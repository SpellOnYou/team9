# -*- coding: utf-8 -*-
# Author: Jiwon Kim, Lara Grimminger

# load standard library
from pathlib import Path
import numpy as np

# import modules needed for MLP
from DNN import DNN_Model

# import text preprocessing modules
from data_preprocessing.one_hot_encoding import OneHotEncoding
from data_preprocessing.input_tf_idf import GenerateSentences, TfIdf
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD


class Main_DNN:

    def __init__(self, **kwargs):
        self.kwargs = {k: v for k, v in kwargs.items()}

        self.is_trace = self.kwargs['trace'] if 'trace' in self.kwargs else False

        # assign n_layer when user input exists
        # self.n_layers = self.kwargs['n_layers'] if 'n_layers' in self.kwargs else 2

    def __call__(self, epochs=10, bs=16, lr=0.025, opt=Adam):

        self.epochs = epochs
        self.bs = bs
        self.lr = lr
        self.opt = opt

        self._get_path()

        if self.is_trace: print("Loading dataset....")
        self._get_data(3000)

        # if self.is_trace: print("Initializing parameters....")
        # self._get_parameter()

        self.dnn = DNN_Model(bs=bs_var, lr=lr_var, opt=opt_var)

        self.train()

        # self.evaluate()

    def _get_path(self):
        """Get data path, as we wrote down in self._get_user_path, this isn't currently user-interactive.
        """

        if 'root_data' in self.kwargs:
            self._get_user_path(self.kwargs['root_path'])

        else:
            self.data_path = Path('datasets/emotions/isear')
            self.train_path = self.data_path / 'isear-train-modified.csv'
            self.valid_path = self.data_path / 'isear-val-modified.csv'
            self.test_path = self.data_path / 'isear-test-modified.csv'

            #path for saving results
            self.eval_path = Path('advanced_model/eval_results')
            assert self.eval_path.is_exist(), "Path for saving results dones't exist"
    def _get_data(self, max_len):
        """This function initializes dataset.

        Note
        -----------
        Since we save gradient to input data as well as our parameters, make sure run this function as well as
         `_init_parameters` if you want to tune hyperparameter.

        """

        # create instances of class GenerateSentences to get x data
        train_sen = GenerateSentences(self.train_path)
        val_sen = GenerateSentences(self.valid_path)
        test_sen = GenerateSentences(self.test_path)

        # create instance of class Tfidf with train dataset
        tfidf_train = TfIdf(train_sen.text)

        tf_idf_train = tfidf_train.tf_idf(train_sen.text)[:, :max_len]  # tf-idf features of train data set
        tf_idf_val = tfidf_train.tf_idf(val_sen.text, train=False)[:,
                     :max_len]  # tf-idf features of validation data set
        tf_idf_test = tfidf_train.tf_idf(test_sen.text, train=False)[:, :max_len]  # tf-idf features of test data set

        # type conversion from numpy float64 to Tensorflow Tensor
        self.x_train = tf.convert_to_tensor(tf_idf_train, dtype=np.float32)
        self.x_valid = tf.convert_to_tensor(tf_idf_val, dtype=np.float32)
        self.x_test = tf.convert_to_tensor(tf_idf_test, dtype=np.float32)
        # x_train, x_valid, x_test = map(np.float32, (tf_idf_train, tf_idf_val, tf_idf_test))
        # self.x_train, self.x_valid, self.x_test = map(Tensor, (x_train, x_valid, x_test))

        # create instances of class OneHotEncoding to get y data
        ohe_train = OneHotEncoding(self.train_path)
        ohe_val = OneHotEncoding(self.valid_path)
        ohe_test = OneHotEncoding(self.test_path)

        reference_dict = ohe_train.get_encoded_dict()  # universal dictionary generated with train data set
        # self.y_train, self.y_valid, self.y_test = map(Tensor, (
        #     ohe_train.one_hot_encoding(),  # one hot encoded emotions of train data set
        #     ohe_val.one_hot_encoding(reference_dict),  # one hot encoded emotions of validation data set
        #     ohe_test.one_hot_encoding(reference_dict)))  # one hot encoded emotions of test data set
        #
        self.y_train = tf.convert_to_tensor(ohe_train.one_hot_encoding(), dtype=np.float32)
        self.y_valid = tf.convert_to_tensor(ohe_val.one_hot_encoding(reference_dict), dtype=np.float32)
        self.y_test = tf.convert_to_tensor(ohe_test.one_hot_encoding(reference_dict), dtype=np.float32)

        # save label to index & index to label for later use in evaluation
        self.label2idx = {label: ohe.index(1) for label, ohe in reference_dict.items()}
        self.idx2label = {idx: label for label, idx in self.label2idx.items()}

    def _get_user_path(self, path):
        user_path = Path(path)
        assert user_path.is_dir(), "root of data should be folder"

    def train(self):

        # dnn = DNN_Model()
        model = self.dnn.define_model()
        trained_model = self.dnn.compile_fit_model(model, self.x_train, self.y_train, self.x_valid, self.y_valid)
        cr = self.dnn.evaluate(model, self.x_test, self.y_test)
        # print(cr)


if __name__ == "__main__":
    # get keyward parameters (if exists)
    dnn_model = Main_DNN()

    bs = [16, 32, 64]
    lr = [0.0001, 0.0005, 0.001, 0.005, 0.01]
    opt = [Adam, SGD]

    for opt_var in opt:
        for bs_var in bs:
            for lr_var in lr:
                dnn_model(bs=bs_var, lr=lr_var, opt=opt_var)
