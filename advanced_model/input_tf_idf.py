# Author: Lara Grimminger

# Load libraries
import numpy as np
import csv


class GenerateSentences:
    """
    A class used to return the input data in a list

    Attributes
    ----------
    text: {list}
        Contains the sentences of the file

    Method
    ----------
    get_sentences
        Returns list of sentences

    """
    def __init__(self, file_name):
        """
            Reads the file which is in csv format into a dictionary form.
            Afterwards it appends the input data of the column "text" to a list.

            Parameter
            ----------
            file_name: {str}
                the name of the csv file
            """
        csv_file = csv.DictReader(open(file_name))
        self.text = []
        for col in csv_file:
            self.text.append(col["text"]) # column "text" of the respective data set

    def get_sentences(self):
        """
        Returns list of sentences

        """
        return self.text


class TfIdf:

    """
    A class used to convert  the input text data into numerical data with tf-idf approach.

    Attributes
    ----------
    punctuations: {str}
        Punctuation marks

    upper: {str}
        Upper case alphabet

    stopwords: {list}
        List of stopwords from NLTK

    vocab: {list}
        List of vocabulary of the train data set

    dict_idx: {dict}
        Dictionary which contains the index of the words of the vocabulary

    word_count: {dict}
        Dictionary which contains the count of the words of the vocabulary

    N_sentences: {int}
        Total numbers of sentences in file

    idf_train: {dict}
        dictionary of inverse document frequency of the train data set

    Methods
    -------
    lowercase_tokenize(sentences):
        Receives a list of sentences as parameter.
        Loops over every sentence in the list and every token in the sentence and replaces uppercase with lowercase.
        Tokenizes list of lowercase sentences
        Returns a list of lowercase and tokenized sentences

    remove_stopwords(tokenized_sentences):
        Receives the lowercase and tokenized list of sentences as parameters.
        Loops over every token in this list and removes stopwords.
        Returns a list of tokens that does not contain any stop word.

    generate_vocabulary(sentences):
        Receives a list of sentences as parameter.
        Loops over every sentence in the list and calls the methods lowercase_tokenize(sentences) and
        remove_stopwords(tokenized_sentences).
        Loops over every token in the lowercase, stop word filtered and tokenized list of sentences and
        appends all unique words to a vocabulary list.
        Returns the vocabulary.

    indexing:
        Loops over every word of the vocabulary and assigns an index to each word.
        Returns a dictionary which contains the vocabulary and its index.

    count_dictionary(input_sentences):
        Counts the frequency of each term of the vocabulary in the data set.
        Returns a dictionary which contains the vocabulary and the frequency of the individual terms.

     term_freq(sentence, word):
        Receives sentence and word as parameters.
        Calculates the number of times a term appears in a sentence.
        Returns the term frequency of each term of each sentence.

    inverse_doc_freq(word):
        Receives word as parameter.
        Computes the document frequency which is the number of sentences a term occurs in.
        Calculates the inverse document frequency by taking the log base 10 of the the number of sentences
        in the data set divided by the document frequency.
        Returns the inverse document frequency.

    tf_idf(input_sentences, train=true):
        Receives the input data as parameter.
        Loops over every sentence in the list and calls the methods lowercase_tokenize(sentences) and
        remove_stopwords(tokenized_sentences).
        Loops over every token in the lowercase, stop word filtered and tokenized list of sentences and
        calls the methods term_freq(word) and inverse_doc_freq(word).
        Calculates tf-idf.
        Returns a matrix of shape (Number of sentences in data set, tf-idf weighted features)
    """
    def __init__(self, list_of_sentences):

        """
        Parameter
        ----------
        list_of_sentences: {list}
            List of input sentences

        """
        self.punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~''' # Punctuation marks
        self.upper = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' # Upper case alphabet
        self.stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",
                          "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
                          'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them',
                          'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll",
                          'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
                          'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
                          'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
                          'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from',
                          'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
                          'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
                          'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
                          'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now',
                          'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn',
                          "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't",
                          'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't",
                          'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won',
                          "won't", 'wouldn', "wouldn't"] # NLTK stopwords list

        self.vocab = self.generate_vocabulary(list_of_sentences)  # Generates the vocabulary
        self.dict_idx = self.indexing()  # Generates the indexing
        self.word_count = self.count_dictionary(list_of_sentences) # Generates the word cound
        self.N_sentences = len(list_of_sentences) # Total number of sentences in file
        self.idf_train = {} # Dictionary of inverse document frequency of the train data set

    def lowercase_tokenize(self, sentences):
        """
        Returns a list of lowercase tokens.

        Parameter
        ----------
        sentences:  {list}
            List of sentences

        """
        lowercase = ""
        for char in sentences:
            if char in self.upper:
                unicode_char_caps = ord(char) # Returns the unicode number of the character
                lower = unicode_char_caps + 32 # Adds 32 to unicode number to get the lowercase character
                lowercase += (chr(lower)) # Converts numerical representation of character to textual representation and
                                          # adds character to the string
            elif char in self.punctuations:
                continue
            else:
                lowercase += char
        lowercase = lowercase.strip()
        tokenized = list(lowercase.split()) # Tokenizing
        return tokenized

    def remove_stopwords(self, tokenized_sentences):

        """
        Returns a list of tokens from which stop words have been removed.

        Parameter
        ----------
        tokenized_sentences: {list}
            List of tokens

        """
        filtered_list = []
        for token in tokenized_sentences:
            if token in self.stopwords:
                continue
            else:
                filtered_list.append(token)
        return filtered_list

    def generate_vocabulary(self, sentences):

        """
        Returns the vocabulary of the train data set.

        Parameter
        ----------
        sentences: {list}
            List of sentences

        """
        vocab = []
        for sentence in sentences:
            tokenized_sentence = self.lowercase_tokenize(sentence)
            filtered_tokenized_sentence = self.remove_stopwords(tokenized_sentence)
            for word in filtered_tokenized_sentence:  # Append only unique words
                if word not in vocab:
                    vocab.append(word)
        return vocab

    def indexing(self):
        """
        Returns a dictionary which contains the index for every word in the vocabulary.

        """
        index_word = {}
        for word, index in zip(self.vocab, range(len(self.vocab))):
            index_word[word] = index
        return index_word

    def count_dictionary(self, input_sentences):
        """
        Returns a dictionary which contains the frequency of each term of the vocabulary in the sentences of the file.

        Parameter
        ----------
        input_sentences: {list}
            List of sentences

        """
        term_count = {}
        for term in self.vocab:
            term_count[term] = 0.0
            for sentence in input_sentences:
                if term in sentence:
                    term_count[term] += 1.0
        return term_count

    def term_freq(self, sentence, word):
        """
        Returns the number of times a term appears in a sentence.

        Parameters
        ----------
        sentence: {str}
            The input sentence
        word: {str}
            The word of an input sentence

        """
        sentence_length = float(len(sentence))
        freq = 0.0
        for term in sentence:
            if term == word:
                freq += 1.0

        return freq / sentence_length

    def inverse_doc_freq(self, word):
        """
        Returns the inverse document frequency of each term that occurs in the train data set.
        Formula: log10(N/df(d,t)+1)

        Parameter
        ----------
        word: {str}
            The word of an input sentence

        Raises
        ----------
        KexError:
            If the term does not occur in the train data set then the term occurrence is assumed to be 1

        """
        try:
            return np.log(self.N_sentences / (self.word_count[word] + 1.0)) # add 1 to avoid a division by 0

        except KeyError:
            return np.log(self.N_sentences) # If term is not in the vocabulary of the train data set then the occurrence
                                            # of that term is assumed ot be 1

    def tf_idf(self, input_sentences, train=True):
        """
        Returns a matrix of tf-idf features
        Formula: tf * idf

        Parameters
        ----------
        input_sentences: {list}
            List of input sentences

        train: {boolean}, default = True
            If the argument is False, then the input sentences are not from the train data set and
            the inverse document frequency is not calculated.
        """
        row = 0
        tf_idf_vec = np.zeros(((len(input_sentences)), (len(self.vocab))))

        for sentence in input_sentences:
            tokenized_sentence = self.lowercase_tokenize(sentence)
            filtered_tokenized_sentence = self.remove_stopwords(tokenized_sentence)
            for word in filtered_tokenized_sentence:
                tf = self.term_freq(filtered_tokenized_sentence, word)
                if train:
                    idf = self.inverse_doc_freq(word)
                    self.idf_train[word] = idf
                    tf_idf = tf * idf
                else:
                    try:
                        tf_idf = tf * self.idf_train[word] # Inverse document frequency of the train data set
                    except KeyError: # If term is not in the vocabulary of the train data set then it is ignored
                        continue

                tf_idf_vec[row][self.dict_idx[word]] = tf_idf

            row += 1

        return tf_idf_vec


# train_file = "../datasets/emotions/isear/isear-train-modified.csv"
# val_file = "../datasets/emotions/isear/isear-val-modified.csv"
# test_file = "../datasets/emotions/isear/isear-test-modified.csv"
#
# pml_train = GenerateSentences(train_file)
# pml_val = GenerateSentences(val_file)
# pml_test = GenerateSentences(test_file)
#
# tfidf = TfIdf(pml_train.get_sentences())
#
# tf_idf_train = tfidf.tf_idf(pml_train.get_sentences())
# tf_idf_val = tfidf.tf_idf(pml_val.get_sentences(), train=False)
# tf_idf_test = tfidf.tf_idf(pml_test.get_sentences(), train=False)
#
# print("tf_idf_train \n", tf_idf_train.shape)
# print(tf_idf_train)
# print("\n")
# print("tf_idf_val \n", tf_idf_val.shape)
# print(tf_idf_val)
# print("\n")
# print("tf_idf_test \n", tf_idf_test.shape)
# print(tf_idf_test)
