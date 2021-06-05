# -*- coding: utf-8 -*-
# Author: Jiwon Kim, Lara Grimminger

# load standard library
from pathlib import Path
from torch import tensor, randn, zeros, no_grad
import numpy as np
import sys
import math

# import modules needed for MLP
from models.model import Model
from metrics.fscore import Fscore

# import text preprocessing modules
from data_preprocessing.one_hot_encoding import OneHotEncoding
from data_preprocessing.input_tf_idf import GenerateSentences, TfIdf


class MLP:
    """Make fully-connected multilayer perceptron.
    Parameters
    ----------
    n_layers : {int}, default = 2
        Number of layers in neural networks. Dimension of each parameter is pre-defined.

    trace : {bool}, default = False
        Print out current status while proceeding training model, if True.

    """

    def __init__(self, **kwargs):
        self.kwargs = {k: v for k, v in kwargs.items()}

        self.is_trace = self.kwargs['trace'] if 'trace' in self.kwargs else False

        # assign n_layer when user input exists
        self.n_layers = self.kwargs['n_layers'] if 'n_layers' in self.kwargs else 2

    def __call__(self):

        self._get_path()

        if self.is_trace: print("Loading dataset....")
        self._get_data(3000)

        if self.is_trace: print("Initializing parameters....")
        self._get_parameter()

        self.model = Model(self.n_layers, *self.params)

        self.train(epochs=5, bs=16, lr=0.025) 

        self.evaluate()

    def _get_parameter(self):
        """Make parameter list to be yielded to model

        Note
        ----------
        Here we initialized our parameter with random variable.
        However, since our model uses vanila architecture which is verneralble to weight exploding/vanishing,
        used a small trick known for kaiming initialization.
        For more information, please see https://arxiv.org/pdf/1502.01852.pdf section 2.2
        """

        # TODO: Also we can use __setattr__ to enable model to approach parameters of layer.

        n_features = self.x_train.shape[1]
        n_class = self.y_train.shape[1]

        self.params = []
        self.layer2param = {2: [n_features, 100, n_class], 4: [n_features, 145, 32, 15, n_class]}

        assert max(self.layer2param.keys()) > self.n_layers, "Your input layer size exceeds our budget."

        # define parameter given layers

        for layer_i in range(self.n_layers):
            # get input, output feature length
            in_f = self.layer2param[self.n_layers][layer_i]
            out_f = self.layer2param[self.n_layers][layer_i + 1]

            self.params += [randn(in_f, out_f) / math.sqrt(out_f), randn(out_f)]

    def train(self, epochs, bs, lr):
        self.epochs, self.bs, self.lr = epochs, bs, lr
        """
        Args:
        """

        for e in range(epochs):

            for bs_i in range((self.x_train.shape[0] - 1) // bs + 1):

                str_idx, end_idx = bs_i * bs, (bs_i + 1) * bs

                x_batch, y_batch = self.x_train[str_idx:end_idx], self.y_train[str_idx:end_idx]

                prediction = self.model.forward(x_batch)

                loss = self.model.loss(prediction, y_batch)

                self.model.backward()

                with no_grad():
                    for l_i, layer in enumerate(self.model.layers):
                        if hasattr(layer, 'w'):  # if they have parameter attribute

                            layer.w -= layer.w.g * lr
                            layer.b -= layer.b.g * lr
                            layer.w.g.zero_()  # initialize them to zero
                            layer.b.g.zero_()

                            # this statistics will be helpful when we track the parameter status
                            if self.is_trace:
                                tot_w_mean = layer.w.mean()
                                tot_w_std = layer.w.std()

                if self.is_trace and (bs_i % 10 == 0 and bs_i):
                    print(f"batch size:{self.bs}, {bs_i}th batch training is done.")
                    print(f"mean weight of batches: {tot_w_mean / 10}, std weight of batches: {tot_w_std / 10}")
                    tot_w_mean, tot_w_std = 0, 0

    def evaluate(self):
        # get output of valid data from our trained model
        pred_valid = self.model.forward(self.x_valid)

        loss_valid = self.model.loss(pred_valid, self.y_valid)

        # get results of softmax(x) to calculate fscore
        softmax_pred = self.model.loss.log_softmax(pred_valid)
        measure = Fscore(softmax_pred, self.y_valid)
        p, r, f = measure()

        # restrict floating point to 3
        precision = [str(value)[:5] for value in p]
        recall = [f"{value:.3f}" for value in r]
        fscore = [f"{value:.3f}" for value in f]

        labels = list(self.label2idx.keys())

        print('\t\t\t\t' + '\t\t'.join(labels[0:2]) + '\t'+ '\t'.join(labels[2:7]),
              'precision:\t\t' + '\t'.join(precision), 'recall:\t\t\t' + '\t'.join(recall),
              'fscore(a=.5):\t' + '\t'.join(fscore), sep='\n')

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

        # type conversion from numpy float64 to pytorch float32
        x_train, x_valid, x_test = map(np.float32, (tf_idf_train, tf_idf_val, tf_idf_test))
        self.x_train, self.x_valid, self.x_test = map(tensor, (x_train, x_valid, x_test))

        # create instances of class OneHotEncoding to get y data
        ohe_train = OneHotEncoding(self.train_path)
        ohe_val = OneHotEncoding(self.valid_path)
        ohe_test = OneHotEncoding(self.test_path)

        reference_dict = ohe_train.get_encoded_dict()  # universal dictionary generated with train data set
        self.y_train, self.y_valid, self.y_test = map(tensor, (
            ohe_train.one_hot_encoding(),  # one hot encoded emotions of train data set
            ohe_val.one_hot_encoding(reference_dict),  # one hot encoded emotions of validation data set
            ohe_test.one_hot_encoding(reference_dict)))  # one hot encoded emotions of test data set

        # save label to index & index to label for later use in evaluation
        self.label2idx = {label: ohe.index(1) for label, ohe in reference_dict.items()}
        self.idx2label = {idx: label for label, idx in self.label2idx.items()}

    def _get_user_path(self, path):
        user_path = Path(path)
        assert user_path.is_dir(), "root of data should be folder"

    # Now we suppose use input is confined to ISEAR dataset. Since our model isn't flexible for various input.
    # Not implemented yet but could be used (e.g. model test)


if __name__ == "__main__":
    # get keyward parameters (if exists)
    mlp = MLP(**dict(arg.split('=') for arg in sys.argv[1:]))

    mlp()
