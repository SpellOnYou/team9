
# load libraries
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk import word_tokenize, pos_tag
import pandas as pd
import nltk

nltk.download('sentiwordnet')


class OverallSentencePolarity:
    """
    A class used to generate the osp values directly from the text.

    Attributes
    ----------
    files_for_train: {dict}
        Dictionary which contains the original train data set and the new data set with the osp column and values

    files_for_val:  {dict}
         Dictionary which contains the original validation data set and the new data set with the osp column and values

    files_for_test:  {dict}
         Dictionary which contains the original test data set and the new data set with the osp column and values

    files: {dict}
        Nested dictionary of train, validation and test set

    file_key:
        Indicates train, validation and test set

    tagged_sentence: {list}
        List of POS-tagged sentences

    sentence: {string}

    score_indexes: {list}
        List of scores

    csv_file: {dataframe}
        Pandas dataframe of original train, val and test files

    isear_text: {dataframe}
        Pandas dataframe of column text of original train, val and test files

    Methods
    -------
    calculate_osp:
        Loops over each sentence of column text in the respective dataframe
        Calls method pos_tag after tokenizing the respective sentence
        Adds POS tags to each sentence
        Calls method get_polarity_score
        Calls method write_score_to_csv

    get_polarity_score:
        Loops over each word and POS-tag
        Calls method get_wordnet_pos(tag) to map Treebank tags to WordNet
        Lemmatizes each word of a sentence
        Creates the synset for each word of a sentence
        Loops over each sense in the synset
        Sums up the positive and negative score of each sense of each word
        The biggest score in a sentence decided whether score is positive or negative
        If there is almost no difference between positive and negative score, the sentene is neutral

    get_wordnet_pos:
        Maps Treebank tags to WordNet tags

    write_score_to_csv:
        Writes osp values in column osp in original data files

    """

    def __init__(self, file_key):

        """

        Parameters
        ----------
        file_key: indicator for train, validation and test set
        """

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

        """
        Tokenizes and POS-tags the sentences
        Calls method get_polarity_score to get osp values
        Writes scores to respective csv files

        """
        for sentence in self.isear_text:
            self.sentence = sentence
            self.tagged_sentence = pos_tag(word_tokenize(sentence))
            self.get_polarity_score()
        self.write_score_to_csv()

    def get_polarity_score(self):

        """
        Gets the polarity score of each sense of each synset of each word
        Calculates the average of the scores
        Biggest scores indicates whether polarity of the sentence is positive or negative
        If there is almost no difference between positive and negative score, the sentence is neutral

        Returns osp values
        -------

        """
        lemmatizer = WordNetLemmatizer()

        pos = 0
        neg = 0
        neutral = 0

        for word, tag in self.tagged_sentence:
            pos_word = 0
            neg_word = 0

            wn_tag = self.get_wordnet_pos(tag)
            if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV, wn.VERB): # If Treebank tag can not be mapped to WordNet, skip
                continue

            lemma = lemmatizer.lemmatize(word, pos=wn_tag)
            if not lemma:
                continue

            synsets = wn.synsets(lemma, pos=wn_tag)
            if not synsets:
                continue

            for synset in synsets:
                swn_synset = swn.senti_synset(synset.name()) # contains scores
                pos_word += swn_synset.pos_score() # retrieves positive score
                neg_word += swn_synset.neg_score() # retrieves negative score

            avg_pos = pos_word / len(synsets) # average for positive score
            avg_neg = neg_word / len(synsets) # average for negative score

            pos += avg_pos # add positive scores of the whole sentence
            neg += avg_neg # add negative scores of the whole sentence

        length_of_sentence = len(self.tagged_sentence)
        scores = [pos / length_of_sentence, neg / length_of_sentence, neutral] # average for the sentence
        max_index = scores.index(max(scores)) # looks for biggest score
        if abs(scores[0] - scores[1]) < 0.001:  # If difference between positive and negative score is smaller than 1%,
                                                # the score is neutral
            max_index = 2
        self.score_indexes.append(max_index)

    def get_wordnet_pos(self, treebank_tag):


        """

        Maps treebank tags to WordNet

        Parameter
        ----------
        treebank_tag

        Returns the respective WordNet tags
        -------

        """

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

        """

        Writes respective osp values in new column osp to the respective csv file

        """
        df = self.csv_file
        osp = self.score_indexes
        df['osp'] = osp # new column
        df.to_csv(self.files[self.file_key]["occ_osp"])


# Main method

# Generates respective train, validation and test instance of class OverallSentencePolarity
osp_train = OverallSentencePolarity("train")
osp_val = OverallSentencePolarity("val")
osp_test = OverallSentencePolarity("test")

# Calls method calculate_osp
osp_train.calculate_osp()
osp_val.calculate_osp()
osp_test.calculate_osp()
