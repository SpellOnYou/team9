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

    def __init__(self, list_of_sentences):

        # define punctuation and upper case alphabet
        self.punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
        self.upper = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
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
                          'd',
                          'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn',
                          "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't",
                          'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't",
                          'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won',
                          "won't", 'wouldn', "wouldn't"]
        self.vocab = self.generate_vocabulary(list_of_sentences)  # Generate the vocabulary
        # print(len(self.vocab))
        self.dict_idx = self.indexing(self.vocab)  # Generate the indexing
        self.word_count = self.count_dictionary(list_of_sentences)
        self.N_sentences = len(list_of_sentences)
        self.idf_train = {}

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

    def remove_stopwords(self, tokenized_sentences):
        filtered_list = []
        for token in tokenized_sentences:
            if token in self.stopwords:
                continue
            else:
                filtered_list.append(token)
        return filtered_list

    def generate_vocabulary(self, all_padded_sentences):
        vocab = []
        for sentence in all_padded_sentences:
            tokenized_sentence = self.lowercase_tokenize(sentence)
            filtered_tokenized_sentence = self.remove_stopwords(tokenized_sentence)
            for word in filtered_tokenized_sentence:  # append only unique words
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
        sentence_length = float(len(sentence))
        occurrence = float(len([token for token in sentence if token == word]))
        return occurrence / sentence_length

    def inverse_doc_freq(self, word):
        try:
            word_occurrence = self.word_count[word] + 1.0
        except KeyError:
            word_occurrence = 1.0
        return np.log(self.N_sentences / word_occurrence)

    def tf_idf(self, input_sentences, train=True):
        row = 0
        tf_idf_vec = np.zeros(((len(input_sentences)), (len(self.vocab))))

        for sentence in input_sentences:
            tokenized_sentence = self.lowercase_tokenize(sentence)
            filtered_tokenized_sentence = self.remove_stopwords(tokenized_sentence)
            for word in filtered_tokenized_sentence:
                tf = self.termfreq(filtered_tokenized_sentence, word)
                if train:
                    idf = self.inverse_doc_freq(word)
                    self.idf_train[word] = idf
                    value = tf * idf
                else:
                    try:
                        value = tf * self.idf_train[word]
                    except KeyError:
                        continue

                tf_idf_vec[row][self.dict_idx[word]] = value

            row += 1

        return tf_idf_vec
