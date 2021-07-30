# dataset.py

from .base import (
    tfidf,
    load_csv,
    load_npz,
    load_pkl,
    validate_y
    )

import numpy as np

# emb_type = {'pretrained': ['fasttext'], 'count': ['tfidf']}
crt_emb_type = ['fasttext','tfidf']

class DataBunch():
    """load all data and convert it to appropriate embedding type. embedding type will be obtrained in main(classify) module

    Parameters
    ----------

    Note: when you load embedding from pre-trained model, it will have tensor size (dataset, max_length, emb_dim) which can't feed on MLP/NB.
        It will be flattend (i.e. tensor size (dataset, max_length * emb_dim)) unless other option is given.
    """ 
    def get_embedding(self, emb_type, model_type, dim, **kwargs):
        #TODO: it's better to divide fit / transform of tfidf and override function with additional class,
        # so that both type of embedding can be obtained same method

        assert emb_type in crt_emb_type, f"You've given currently unavailable embedding option : {self.kwargs['emb_type']},\nWe support {crt_emb_type} only."

        # when embeddig type is count based
        if emb_type=='tfidf':
            self.embedder = tfidf()
            self.x_train = self.embedder.fit_transform(self.x_train_text).toarray()
            self.x_test = self.embedder.transform(self.x_test_text).toarray()

        #when it's pretrained version
        else:
            self.embedder = self._load_pretrained(emb_type, dim)
            xt = np.array([self.embedder[pad_sent, ] for pad_sent in self.train_pad_nums])
            xv = np.array([self.embedder[pad_sent, ] for pad_sent in self.test_pad_nums])
            self.x_train = xt.reshape(xt.shape[0], -1)
            self.x_test = xv.reshape(xv.shape[0], -1)
            # print(self.x_train.shape, self.x_test.shape)
            # make y to one-hot

        if model_type.lower()=='mlp':
            self.y_train, self.y_test = map(self.onehot, (self.y_train, self.y_test))

    def onehot(self, labels):
        """
        In case of neural network, we need to expand label to one-hot encoded matrix
        """
        shape = (len(labels), max(labels)+1)
        
        mat = np.zeros(shape=shape)
        mat[range(shape[0]), labels] = 1
        return mat

    def get_data(self, occ):
        """
        A function which loads dataset with given (or default) file path

        Attribute
        x_train_text: (list(one data))
            list of text data, and each sentence(or document) is not tokenized
        y_test_label: (list(one data))
        """
        #TODO:we can get dir path's data at once and split by ourselves

        (xt, yt) = load_csv(fpath = 'example/train', occ_type=occ)
        (xv, yv) = load_csv(fpath = 'example/valid', occ_type=occ)
        self.x_train_text = xt + xv
        self.y_train_label = yt + yv

        self.x_test_text, self.y_test_label = load_csv(fpath='example/test', occ_type = occ)
        
        # change when labels aren't int with saving label's index
        self.label2idx, y_train = validate_y(self.y_train_label)
        self.c = len(self.label2idx) #number of classes
        self.labels = list(self.label2idx.keys())
        y_test = list(map(lambda x: self.label2idx[x], self.y_test_label))
        self.y_train, self.y_test = map(np.array, (y_train, y_test))

    def _load_pretrained(self, emb_type, dim, token='word', **kwargs):
        """This function will 
            1. Tokenize according to tokenization type
                1.a. when it's word it uses tfidf vectorizer's tokenizer
                    : Here we use same tokenizer which we used in tfidf, or other counter-based embedding
                1.b. when it's chracter it will use null space as delimeter
                Note: Other types of tokenizing (i.e., hybrid, byte, etc) vectors are not implemented yet
            2. Map tokenized words to indices of embedding and padding (if max_seq is given, sentence will be processed on that lenght)
            3. Rretrieve pretrained embedding
        """
        kwargs = {k:v for k, v in kwargs.items()}
        #first process
        tokenizer = tfidf().build_tokenizer()
        train_tokens = list(map(tokenizer, self.x_train_text))
        test_tokens = list(map(tokenizer, self.x_test_text))

        #second process
        #mapping to index
        self.vtoi = load_pkl(emb_type)
        num_fn = lambda tokens: [self.vtoi[token.lower()] if token.lower() in self.vtoi else -1 for token in tokens ]
        
        train_numeric = map(num_fn, train_tokens)
        test_numeric = map(num_fn, test_tokens)

        #padding, vacant words will be substituted as unknown
        if not 'max_seq' in kwargs: max_seq = max([len(i) for i in train_tokens])
        # this can be used for later (i.e., evaluating)
        if not hasattr(self, 'max_seq'): self.max_seq = max_seq
        pad_fn = lambda sent: sent + [-1] * (max_seq-len(sent))
        self.train_pad_nums = list(map(pad_fn, train_numeric))
        self.test_pad_nums = list(map(pad_fn, test_numeric))
        
        #third process
        pre_trained = load_npz(dim, emb_type)
        return pre_trained
