import numpy as np
import csv


class PadMaxLength:

    def __init__(self, file_name):
        self.file = open(file_name)
        self.csv_file = csv.DictReader(self.file)
        self.list_padded_sentences = []
        self.text = []
        for col in self.csv_file:
            self.text.append(col["text"])

    def min_max_sentences(self):
        tokenized_sentences = []
        # split each sentence into words
        for sentence in self.text:
            tokens = sentence.split()
            tokenized_sentences.append(tokens)
        # get longest sentence and its length
        longest_sent = max(tokenized_sentences, key=len)
        longest_sent_len = len(longest_sent)

        # get shortest word and its length
        shortest_sent = min(tokenized_sentences, key=len)
        shortest_sent_len = len(shortest_sent)

        return longest_sent_len, shortest_sent_len

    def right_pad_sentences(self, max_sent_length):
        max_len = round(max_sent_length * 0.50)  # Take 50% of the maximum sentence length to avoid sparsity
        padded_sentences = []
        # print(max_len)

        for sentence in self.text:
            sentence = sentence.strip()
            sentence = sentence.split()

            if len(sentence) > max_len:
                a = sentence[:max_len]  # discard tokens longer than max_length
                padded_sentences.append(a)

            elif len(sentence) < max_len:
                [sentence.append("0") for i in
                 range(max_len - len(sentence))]  # pad sentences with zeros smaller than max_length
                padded_sentences.append(sentence)

            else:
                padded_sentences.append(sentence)

        for pad_sent in padded_sentences:
            list_sentences = ' '.join(pad_sent)
            self.list_padded_sentences.append(list_sentences)

        return self.list_padded_sentences

    def merge_with(self, list2, list3):
        merged = self.list_padded_sentences + list2 + list3
        return merged


class BagOfWords:

    def __init__(self, all_padded_sentences, list_of_sentences):

        # define punctuation and upper case alphabet
        self.punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
        self.upper = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        self.vocab = self.generate_vocabulary(all_padded_sentences)  # Generate the vocabulary
        #print(len(self.vocab))
        self.dict_idx = self.indexing(self.vocab)  # Generate the indexing
        self.word_count = self.count_dictionary(list_of_sentences)
        self.N_sentences = len(list_of_sentences)

    def lowercase_tokenize(self, padded_sentences):
        lowercase = ""
        for char in padded_sentences:
            if char in self.upper:
                k = ord(char)
                l = k + 32
                lowercase = lowercase + (chr(l))
            elif char in self.punctuations:
                continue
            else:
                lowercase = lowercase + char
        lowercase = lowercase.strip()
        tokenized = list(lowercase.split())
        return tokenized

    def generate_vocabulary(self, all_padded_sentences):
        vocab = []
        for sentence in all_padded_sentences:
            tokenized_sentence = self.lowercase_tokenize(sentence)
            for word in tokenized_sentence:  # append only unique words
                if word not in vocab:
                    vocab.append(word)
        return vocab

    def indexing(self, tokens):
        # Index dictionary to assign an index to each word in vocabulary
        index_word = {}
        i = 0
        for word in tokens:
            index_word[word] = i
            i += 1
        return index_word

    def count_dictionary(self, input_sentences):
        word_count = {}
        for word in self.vocab:
            word_count[word] = 0.0
            for sentence in input_sentences:
                if word in sentence:
                    word_count[word] += 1.0
        return word_count

    # Term Frequency
    def termfreq(self, sentence, word):
        number_of_sentences = float(len(sentence))
        occurrence = float(len([token for token in sentence if token == word]))
        return occurrence / number_of_sentences

    def inverse_doc_freq(self, word):
        try:
            word_occurrence = self.word_count[word] + 1.0
        except KeyError:
            word_occurrence = 1.0
        return np.log(self.N_sentences / word_occurrence)

    def tf_idf(self, input_sentences):
        row = 0
        tf_idf_vec = np.zeros((self.N_sentences, (len(self.vocab))))

        for sentence in input_sentences:
            tokenized_sentence = self.lowercase_tokenize(sentence)
            for word in tokenized_sentence:
                tf = self.termfreq(sentence, word)
                idf = self.inverse_doc_freq(word)

                value = tf * idf
                tf_idf_vec[row][self.dict_idx[word]] = value

            row += 1

        return tf_idf_vec

    # def bag_of_words(self, input_sentences):
    #     bow_vector = np.zeros((len(input_sentences), len(self.vocab)))  # Nr of sentences x length of vocabulary
    #     row = 0
    #     for sentence in input_sentences:
    #         tokenized_sentence = self.lowercase_tokenize(sentence)
    #         for word in tokenized_sentence:
    #             #print(word)
    #             bow_vector[row][self.dict_idx[word]] += 1  # Add the occurrence
    #         row += 1
    #     return bow_vector


train_file = "../data/emotions/isear/isear-train-modified.csv"
val_file = "../data/emotions/isear/isear-val-modified.csv"
test_file = "../data/emotions/isear/isear-test-modified.csv"

pml_train = PadMaxLength(train_file)
pml_val = PadMaxLength(val_file)
pml_test = PadMaxLength(test_file)

max_sent, min_sent = pml_train.min_max_sentences()

sentences_padded_train = pml_train.right_pad_sentences(max_sent)
# print("Len first sentence of train File", len(sentences_padded_train[0].split()))

sentences_padded_val = pml_val.right_pad_sentences(max_sent)
# print("Len first sentence of val File", len(sentences_padded_val[0].split()))

sentences_padded_test = pml_test.right_pad_sentences(max_sent)
# print("Len first sentence of test File", len(sentences_padded_test[0].split()))

vocab_list = pml_train.merge_with(sentences_padded_val, sentences_padded_test)  # Vocab over all files

bow_train = BagOfWords(vocab_list, sentences_padded_train)  # Sentences to create the vocabulary
bow_val = BagOfWords(vocab_list, sentences_padded_val)
bow_test = BagOfWords(vocab_list, sentences_padded_test)

tf_idf_train = bow_train.tf_idf(sentences_padded_train)
tf_idf_val = bow_val.tf_idf(sentences_padded_val)
tf_idf_test = bow_test.tf_idf(sentences_padded_test)

print("tf_idf_train \n", tf_idf_train.shape)
print(tf_idf_train)
print("\n")
print("tf_idf_val \n", tf_idf_val.shape)
print(tf_idf_val)
print("\n")
print("tf_idf_test \n", tf_idf_test.shape)
print(tf_idf_test)


# vector_train = bow_train.bag_of_words(sentences_padded_train)  # BOW over some other sentences
# vector_val = bow_val.bag_of_words(sentences_padded_val)  # BOW over some other sentences
# vector_test = bow_test.bag_of_words(sentences_padded_test)  # BOW over some other sentences
#
# print(vector_train.shape)
# print(vector_train)
#
# print(vector_val.shape)
# print(vector_val)
#
# print(vector_test.shape)
# print(vector_test)

