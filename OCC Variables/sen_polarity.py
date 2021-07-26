from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk import word_tokenize, pos_tag
import pandas as pd
import nltk

nltk.download('sentiwordnet')


class OverallSentencePolarity:

    def __init__(self, file_key):
        files_for_train = {"modified": "isear-train-modified-v2.csv", "occ_osp": "train_occ_osp.csv"}
        files_for_valid = {"modified": "isear-valid-modified-v2.csv", "occ_osp": "val_occ_osp.csv"}
        files_for_test = {"modified": "isear-test-modified-v2.csv", "occ_osp": "test_occ_osp.csv"}
        self.files = {"train": files_for_train, "val": files_for_valid, "test": files_for_test}
        self.file_key = file_key
        self.tagged_sentence = []
        self.sentence = ""
        self.score_indexes = []
        self.csv_file = pd.read_csv(self.files[file_key]["modified"])
        self.isear_text = self.csv_file["text"]

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

            # synset = synsets[0]
            for synset in synsets:
                # print(synset)

                swn_synset = swn.senti_synset(synset.name())

                pos_word += swn_synset.pos_score()
                neg_word += swn_synset.neg_score()
                # neutral_word += swn_synset.obj_score()

            avg_pos = pos_word / len(synsets)
            avg_neg = neg_word / len(synsets)
            # avg_neutral = neutral_word / len(synsets)

            pos += avg_pos
            neg += avg_neg
            # neutral += avg_neutral
        length_of_sentence = len(self.tagged_sentence)
        scores = [pos / length_of_sentence, neg / length_of_sentence, neutral]
        max_index = scores.index(max(scores))
        if abs(scores[0] - scores[1]) < 0.001:
            max_index = 2
        if max_index == 2:
            print("Neutral")
            print(scores)
        # elif max_index == 1:
        # print("Negative")
        # elif max_index == 2:
        # print("Neutral")

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
        df = self.csv_file
        osp = self.score_indexes
        df['osp'] = osp
        df.to_csv(self.files[self.file_key]["occ_osp"])


osp_train = OverallSentencePolarity("train")
osp_val = OverallSentencePolarity("val")
osp_test = OverallSentencePolarity("test")

osp_train.calculate_osp()
osp_val.calculate_osp()
osp_test.calculate_osp()
