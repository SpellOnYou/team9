# -*- coding: utf-8 -*-
# Author: Jiwon Kim, Lara Grimminger

# load standard library
from itertools import combinations, product, chain
from pathlib import Path
import numpy as np

# import modules needed for MLP
from dnn_lime import LimeExplainer
#from dnn_model import DnnModel
from second_input import OCCVariables
from multi_inputs_mlp import MultipleInputsModel

# import text preprocessing modules
from data_preprocessing.one_hot_encoding import OneHotEncoding
from data_preprocessing.input_tf_idf import GenerateSentences, TfIdf
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD

CASE_TYPE = ["data", "rule"]


class MainDNN:

    def __init__(self, **kwargs):
        self.kwargs = {k: v for k, v in kwargs.items()}

        self.is_trace = self.kwargs['trace'] if 'trace' in self.kwargs else False

        self.vocab = None
        self.x_train_occ = None
        self.x_test_occ = None
        self.features = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.multi_model = None
        self.type_data = False
        self.type_rule = False

    def __call__(self, epochs=20, bs=64, lr=0.0001, opt=Adam):

        self.epochs = epochs
        self.bs = bs
        self.lr = lr
        self.opt = opt

        if self.is_trace:
           print("Loading dataset....")
        test_input, test_output = self.preprocess_first_input() # For LIME

        self.le = LimeExplainer(bs=self.bs, lr=self.lr, opt=self.opt)
        #self.dnn = DnnModel(bs=self.bs, lr=self.lr, opt=self.opt)
        self.train_occ()


    def init_dnn(self, occ_type=None):
        if occ_type == "data":
            self._set_path_data()
            occ_data = ['tense', 'osp']
            self.features = list(list(combinations(occ_data, i + 1)) for i, _ in enumerate(occ_data))
            self.type_data = True
            self.type_rule = False
            ret_val = True
        elif occ_type == "rule":
            self._set_path_rule()
            occ_rule = ['tense', 'direction', 'polarity']
            self.features = list(list(combinations(occ_rule, i + 1)) for i, _ in enumerate(occ_rule))
            self.type_data = False
            self.type_rule = True
            ret_val = True
        else:
            ret_val = False
        return ret_val

    def _set_path_data(self):


        if 'root_data' in self.kwargs:
            self._get_user_path(self.kwargs['root_path'])

        else:
            self.data_path = Path('../datasets/emotions/isear/text-based')
            self.train_path = self.data_path / 'train_val_osp_tense.csv'
            self.test_path = self.data_path / 'test_osp_tense.csv'

    def _set_path_rule(self):


        if 'root_data' in self.kwargs:
            self._get_user_path(self.kwargs['root_path'])

        else:
            self.data_path = Path('../datasets/emotions/isear/rule-driven')
            self.train_path = self.data_path / 'train_val_occ_rule_v3.csv'
            self.test_path = self.data_path / 'test_occ_rule_v3.csv'

    def preprocess_first_input(self):

        # create instances of class GenerateSentences to get x data
        train_sen = GenerateSentences(self.train_path)
        test_sen = GenerateSentences(self.test_path)

        # create instance of class Tfidf with train dataset
        vocab = TfIdf(train_sen.text)
        self.vocab = vocab

        tf_idf_train = vocab.tf_idf(train_sen.text)  # tf-idf features of train data set
        tf_idf_test = vocab.tf_idf(test_sen.text, train=False)  # tf-idf features of test data set

        # type conversion from numpy float64 to Tensorflow Tensor
        self.x_train = tf.convert_to_tensor(tf_idf_train, dtype=np.float32)
        self.x_test = tf.convert_to_tensor(tf_idf_test, dtype=np.float32)

        # create instances of class OneHotEncoding to get y data
        ohe_train = OneHotEncoding(self.train_path)
        ohe_test = OneHotEncoding(self.test_path)

        reference_dict = ohe_train.get_encoded_dict()  # universal dictionary generated with train data set
        print(reference_dict)

        self.y_train = tf.convert_to_tensor(ohe_train.one_hot_encoding(), dtype=np.float32)
        self.y_test = tf.convert_to_tensor(ohe_test.one_hot_encoding(reference_dict), dtype=np.float32)

        return test_sen.get_sentences(), ohe_test.get_labels()

    def preprocess_second_input(self, feature_input):

        train_occ = OCCVariables(self.train_path) # dataset occ rule and data-based
        test_occ = OCCVariables(self.test_path)

        #train_occ = OCCVariables(self.train_path_data)
        #test_occ = OCCVariables(self.test_path_data)

        if self.type_rule:
            self.x_train_occ = tf.convert_to_tensor(train_occ.preprocess_occ_rule(feature_input), dtype=np.float32)
            self.x_test_occ = tf.convert_to_tensor(test_occ.preprocess_occ_rule(feature_input), dtype=np.float32)
        elif self.type_data:
            self.x_train_occ = tf.convert_to_tensor(train_occ.preprocess_occ_data(feature_input), dtype=np.float32)
            self.x_test_occ = tf.convert_to_tensor(test_occ.preprocess_occ_data(feature_input), dtype=np.float32)

        #self.x_train_occ = tf.convert_to_tensor(train_occ.preprocess_occ_data(), dtype=np.float32)
        #self.x_test_occ = tf.convert_to_tensor(test_occ.preprocess_occ_data(), dtype=np.float32)

    def _get_user_path(self, path):
        user_path = Path(path)
        assert user_path.is_dir(), "root of data should be folder"

    def train_occ(self):
        for feature_list in self.features:
            self.multi_model = MultipleInputsModel(len(feature_list[0]), self.x_train.shape[1], bs=self.bs, lr=self.lr,
                                                   opt=self.opt)

            for feature in feature_list:
                text_occ_model = self.multi_model.define_model()
                print(feature)
                self.preprocess_second_input(feature)
                self.multi_model.fit_model(text_occ_model, self.x_train, self.x_train_occ, self.y_train)
                cr_multi, cm_multi = self.multi_model.evaluate(text_occ_model, self.x_test, self.x_test_occ, self.y_test)
                print(cr_multi, cm_multi)



    # def train_lime(self, test_sentences, ohe_test):
    #     self.le.fit_model(self.x_train, self.y_train)
    #     self.le.lime_exp(test_sentences, self.vocab, ohe_test)


if __name__ == "__main__":
    dnn_model = MainDNN()

    for case in CASE_TYPE:
        if dnn_model.init_dnn(case):
            dnn_model()

