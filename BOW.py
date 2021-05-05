import numpy as np

# Sample text corpus
sentences = ['She loves Pizza, pizza is delicious.', 'She is a good Person.', 'good People are the best.']


class BagOfWords:
    def __init__(self, list_of_sentences):
        # define punctuation and upper case alphabet
        self.punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
        self.upper = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        self.vocab = self.generate_vocabulary(list_of_sentences)    # Generate the vocabulary
        self.dict_idx = self.indexing(self.vocab)                   # Generate the indexing

    def lowercase_tokenize(self, sentence):
        lowercase = ""
        for char in sentence:
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
            for word in tokenized_sentence:     # append only unique words
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

    def bag_of_words(self, input_sentences):
        bow_vector = np.zeros((len(input_sentences), len(self.vocab)))  # Nr of sentences x length of vocabulary
        row = 0
        for sentence in input_sentences:
            tokenized_sentence = self.lowercase_tokenize(sentence)
            for word in tokenized_sentence:
                bow_vector[row][self.dict_idx[word]] += 1       # Add the occurrence
            row += 1
        return bow_vector


bow = BagOfWords(sentences)     # Sentences to create the vocabulary
vector = bow.bag_of_words(sentences)    # BOW over some other sentences

print("sentences: ", sentences)
print(type(vector))
print(vector)
