from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk import sent_tokenize, word_tokenize, pos_tag
import pandas as pd
import csv
import nltk

nltk.download('sentiwordnet')


class OverallSentencePolarity:

    def __init__(self, file_name):
        csv_file = pd.read_csv(file_name)
        self.isear_text = csv_file["text"]
        self.tagged_sentence = []
        self.sentence = ""
        self.score_indexes = []

    def calculate_osp(self):
        for sentence in self.isear_text:
            self.sentence = sentence
            self.tagged_sentence = pos_tag(word_tokenize(sentence))
            self.get_polarity_score()
        self.write_score_to_csv()

    def get_polarity_score(self):
        lemmatizer = WordNetLemmatizer()
        scores = []

        pos = 0
        neg = 0
        neutral = 0

        for word, tag in self.tagged_sentence:
            pos_word = 0
            neg_word = 0
            neutral_word = 0
            wn_tag = self.get_wordnet_pos(tag)
            if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV, wn.VERB):
                continue


            lemma = lemmatizer.lemmatize(word, pos=wn_tag)
            if not lemma:
                continue

            synsets = wn.synsets(lemma, pos=wn_tag)
            if not synsets:
                continue


            #synset = synsets[0]
            for synset in synsets:
                #print(synset)

                swn_synset = swn.senti_synset(synset.name())

                pos_word += swn_synset.pos_score()
                neg_word += swn_synset.neg_score()
                #neutral_word += swn_synset.obj_score()

            avg_pos = pos_word / len(synsets)
            avg_neg = neg_word / len(synsets)
            #avg_neutral = neutral_word / len(synsets)

            pos += avg_pos
            neg += avg_neg
            #neutral += avg_neutral
        length_of_sentence = len(self.tagged_sentence)
        scores = [pos/length_of_sentence, neg/length_of_sentence, neutral]
        max_index = scores.index(max(scores))
        if abs(scores[0] - scores[1]) < 0.001:
            max_index = 2
        if max_index == 2:
            print("Neutral")
            print(scores)
        #elif max_index == 1:
            #print("Negative")
        #elif max_index == 2:
            #print("Neutral")

        self.score_indexes.append(max_index)

    # maps the treebank tags to WordNet part of speech names:
    def get_wordnet_pos(self, treebank_tag):

        if treebank_tag.startswith('J'):
            return wn.ADJ
        elif treebank_tag.startswith('V'):
            return wn.VERB
        elif treebank_tag.startswith('N'):
            return wn.NOUN
        elif treebank_tag.startswith('R'):
            return wn.ADV
        else:
            return

    def write_score_to_csv(self):
        file_train = "/home/lara/PycharmProjects/CLab21_local_workspace/OCC Variables/isear-train-modified-v2.csv"
        #file_val = "/home/lara/PycharmProjects/CLab21_local_workspace/OCC Variables/isear-valid-modified-v2.csv"
        #file_test = "/home/lara/PycharmProjects/CLab21_local_workspace/OCC Variables/isear-test-modified-v2.csv"

        df_train = pd.read_csv(file_train)
        #df_val = pd.read_csv(file_val)
        #df_test = pd.read_csv(file_test)

        # Declare a list that is to be converted into a column
        osr_train = self.score_indexes
        #osr_val = self.score_indexes
        #osr_test = self.score_indexes

        df_train['osp'] = osr_train
        #df_val['osp'] = osr_val
        #df_test['osp'] = osr_test

        df_train.to_csv("/home/lara/PycharmProjects/CLab21_local_workspace/OCC Variables/train_occ_osp.csv")
        #df_val.to_csv("/home/lara/PycharmProjects/CLab21_local_workspace/OCC Variables/val_occ_osp.csv")
        #df_test.to_csv("/home/lara/PycharmProjects/CLab21_local_workspace/OCC Variables/test_occ_osp.csv")


isear_train_data = "/home/lara/PycharmProjects/CLab21_local_workspace/OCC Variables/isear-train-modified-v2.csv"
#isear_val_data = "/home/lara/PycharmProjects/CLab21_local_workspace/OCC Variables/isear-valid-modified-v2.csv"
#isear_test_data = "/home/lara/PycharmProjects/CLab21_local_workspace/OCC Variables/isear-test-modified-v2.csv"

osp_train = OverallSentencePolarity(isear_train_data)
#osp_val = OverallSentencePolarity(isear_val_data)
#osp_test = OverallSentencePolarity(isear_test_data)

osp_train.calculate_osp()
#osp_val.calculate_osp()
#osp_test.calculate_osp()



