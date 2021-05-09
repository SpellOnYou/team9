import numpy as np
import pandas as pd

class PadMaxLength:

    def __init__(self, file_name):
        self.csv_file = pd.read_csv(file_name)
        self.text = self.csv_file["text"]

    def min_max_sentences(self):
        tokenized_sentences = []
        # split each sentence into words
        for sentence in self.text:
            tokens = sentence.split(" ")
            tokenized_sentences.append(tokens)
        # print(tokenized_sentences)
        # get longest sentence and its length
        longest_sent = max(tokenized_sentences, key=len)
        longest_sent_len = len(longest_sent)

        # get shortest word and its length
        shortest_sent = min(tokenized_sentences, key=len)
        shortest_sent_len = len(shortest_sent)

        return (longest_sent_len, shortest_sent_len)

    def right_pad_sentences(self, max_sent_length):
        max_len = round(max_sent_length * 0.80) # Take 80% of the maximum sentence length to avoid sparsity
        #print(max_len)
        padded_sentences = []
        list_padded_sentences = []

        for sentence in self.text:
            sent = sentence.strip()
            sent = sent.split(" ")

            if len(sent) > max_len:
                a = sent[:max_len] # discard tokens longer than max_length
                padded_sentences.append(a)

            elif len(sent) < max_len:
                [sent.append("0") for i in range(max_len - len(sent))] # pad sentences with zeros smaller than max_length
                padded_sentences.append(sent)

            else:
                padded_sentences.append(sent)
        #print(padded_sentences)

        for pad_sent in padded_sentences:
            list_sentences = ' '.join(pad_sent)
            list_padded_sentences.append(list_sentences)
            #y =  list(x)
        #print("list_padded_sentences", list_padded_sentences)

        return list_padded_sentences

class BagOfWords:

    def __init__(self, list_of_sentences, file_name):

        # define punctuation and upper case alphabet
        self.punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
        self.upper = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        self.vocab = self.generate_vocabulary(list_of_sentences)  # Generate the vocabulary
        self.dict_idx = self.indexing(self.vocab)  # Generate the indexing
        #self.csv_file = pd.read_csv(file_name)
        #self.text = self.csv_file["text"]

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
        tokenized = list(lowercase.split(" "))
        return tokenized

    def generate_vocabulary(self, list_of_sentences):
        vocab = []
        for sentence in list_of_sentences:
            tokenized_sentence = self.lowercase_tokenize(sentence)
            for word in tokenized_sentence:  # append only unique words
                if word not in vocab:
                    vocab.append(word)
        #print("vocab", vocab)
        return vocab

    def indexing(self, tokens):
        # Index dictionary to assign an index to each word in vocabulary
        index_word = {}
        i = 0
        for word in tokens:
            index_word[word] = i
            i += 1
        return index_word

    def bag_of_words(self, input_sentences):
        bow_vector = np.zeros((len(input_sentences), len(self.vocab)))  # Nr of sentences x length of vocabulary
        row = 0
        for sentence in input_sentences:
            tokenized_sentence = self.lowercase_tokenize(sentence)
            for word in tokenized_sentence:
                #print(word)
                bow_vector[row][self.dict_idx[word]] += 1  # Add the occurrence
            row += 1
        return bow_vector


train_file = "/home/lara/PycharmProjects/pythonProject/CLab21/data/emotions/isear/isear-train-modified.csv"
val_file = "/home/lara/PycharmProjects/pythonProject/CLab21/data/emotions/isear/isear-val-modified.csv"
test_file = "/home/lara/PycharmProjects/pythonProject/CLab21/data/emotions/isear/isear-test-modified.csv"

pml_train = PadMaxLength(train_file)
pml_val = PadMaxLength(val_file)
pml_test = PadMaxLength(test_file)

max_sent, min_sent = pml_train.min_max_sentences()

sentences_padded_train = pml_train.right_pad_sentences(max_sent)
sentences_padded_val = pml_val.right_pad_sentences(max_sent)
sentences_padded_test = pml_test.right_pad_sentences(max_sent)


bow_train = BagOfWords(sentences_padded_train, train_file)  # Sentences to create the vocabulary
bow_val = BagOfWords(sentences_padded_val, val_file)
bow_test = BagOfWords(sentences_padded_test, test_file)

vector_train = bow_train.bag_of_words(sentences_padded_train)  # BOW over some other sentences
vector_val = bow_val.bag_of_words(sentences_padded_val)  # BOW over some other sentences
vector_test = bow_test.bag_of_words(sentences_padded_test)  # BOW over some other sentences

print(vector_train.shape)
print(vector_train)

print(vector_val.shape)
print(vector_val)

print(vector_test.shape)
print(vector_test)
