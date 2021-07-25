# -*- coding: utf-8 -*-
# Author: Jiwon Kim, Lara Grimminger

import matplotlib.pyplot as plt

from pathlib import Path
from fastai.text.all import *
from collections import Counter
import pandas as pd
from sklearn.metrics import classification_report


class LstmAwd:

    def get_data(self, path):
        """
        get data with given path.
        Be careful since some labels (e.g. disgust) have mixed cases (upper, lower)
        """
        with path.open() as f:
            for line_i, line in enumerate(f):
                if line_i == 0:
                    continue
                label, text, osp, tense = line.split(',', maxsplit=3)
                # label, text = line.split(',', maxsplit=1)
                yield label.lower(), text.strip('\n'), osp.strip('\n'), tense.strip('\n')
                # yield label.lower(), text.strip('\n')


    def get_df(self, train_path, valid_path, test_path):
        """
        this function gets train, valid, test path and return train, valid data to data frame format.
        """

        train_data, valid_data, test_data = map(self.get_data, (train_path, valid_path, test_path))
        train_df, valid_df, test_df = map(pd.DataFrame, (train_data, valid_data, test_data))
        train_df['is_valid'], valid_df['is_valid'] = False, True
        train_df = train_df.append(valid_df)

        return train_df, test_df

    def train(self, train_df, textCol, file_name):
        """
        This function composed of mainly 2 phases.
        1. train 2. interpretation.
        you can customize this function as you need.

        In a phase of training, model is initialiezd using AWD_LSTM architecture, with pre-trained language model
        (with wikipedia) using fastai API.
        more specifically,
            First, it makes data loader,
            Second, it defines model finetune shallow epochs
            Third, with given learning rate
                learning rate is given by learning rate finder, which finds out steepest point of error curve.

        After training, model shows interpretation of its result.
            First, case most confused, actual, predict, number of cases from leftmost.
            Second, it shows most highest loss and the text (loss function is cross entropy)
            Finally, plot confusion matrix
        """

        dls = TextDataLoaders.from_df(train_df, path='.', text_col=textCol, label_col=0, valid_col='is_valid')
        learn = text_classifier_learner(dls, AWD_LSTM, drop_mult=0.5, metrics=accuracy)
        learn.fine_tune(1, 1e-2)
        lr_new = learn.lr_find().valley
        learn.fine_tune(1, lr_new)
        #
        # interp = ClassificationInterpretation.from_learner(learn)
        # interp.most_confused(min_val=3)
        # interp.plot_top_losses(10, figsize=(15, 15))
        # interp.plot_confusion_matrix()
        #
        # plt.savefig(file_name)

        return learn

    def test(self, learn, test_path, columns):

        test_df = pd.read_csv(test_path)
        test_features = test_df[columns]
        test_features = test_features.values.tolist()

        ref_dict = {'anger': 1, 'disgust': 2, 'fear': 3, 'guilt': 4, 'joy': 5, 'sadness': 6, 'shame': 7}
        yhat_list = []

        for text in test_features:
            yhat_list.append(learn.predict(text[0])[1].item())

        get_idx = lambda x: ref_dict[x]
        ytrue_list = list(map(get_idx, test_df['label'].tolist()))

        print(classification_report(ytrue_list, yhat_list, target_names=ref_dict.keys()))

'''
you can predict your test data using model.predict (as sklearn.)

e.g. model.predict('When I understood that I was admitted to the University.') will answer you

>>> ('joy',
 tensor(5),
 tensor([5.9464e-04, 5.7787e-04, 1.3048e-03, 1.5658e-03, 9.9196e-01,
         2.3159e-03, 1.3603e-03]))


that means you have answer 'joy' corresponding to tensor 5 (among 0-6), with '9.9196e-01' probability (which is 99%)
'''
