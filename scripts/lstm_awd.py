# -*- coding: utf-8 -*-
# Author: Jiwon Kim

# before executing this code, if you don't have fastai library, I recommend you to download it first using pip install pip install fastai

from pathlib import Path
from fastai.text.all import *
from collections import Counter
import pandas as pd

def get_data(path):
    '''get data with given path.
    Be careful since some labels (e.g. disgust) have mixed cases (upper, lower)
    '''
    with path.open() as f:
        for line_i, line in enumerate(f):
            if line_i == 0: continue
            label, text = line.split(',', maxsplit=1)
            yield label.lower(), text.strip('\n')train_data = get_data(train_path)


def get_df(train_path, valid_path, test_path):
    '''
    this function gets train, valid, test path and return train, valid data to data frame format.
    '''

    train_data, valid_data, test_data = map(get_data, (train_path, valid_path, test_path))
    train_df, valid_df, test_df = map(pd.DataFrame, (train_data, valid_data, test_data))
    
    train_df['is_valid'], valid_df['is_valid'] = False, True    
    
    train_df = train_df.append(valid_df)

    return train_df, test_df

def train(train_df):
    '''
    This function composed of mainly 2 phases.
    1. train 2. interpretation.
    you can customize this fuction as you need.

    In a phase of training, model is initialiezd using AWD_LSTM architecture, with pre-trained language model (with wikipedia) using fastai API.
    more specifically,
        First, it makes data loader,
        Second, it defines model finetune shallow epochs
        Third, with given learning rate
            learning rate is given by learning rate finder, which finds out steepest point of error curve.
        
    After training, model shows interpretation of its result.
        First, case most confused, actual, predict, number of cases from leftmost.
        Second, it shows most highest loss and the text (loss function is cross entropy)
        Finally, plot confusion matrix
    '''

    dls = TextDataLoaders.from_df(train_df, path='.', text_col = 1, label_col = 0, valid_col='is_valid')
    learn = text_classifier_learner(dls, AWD_LSTM, drop_mult=0.5, metrics=accuracy)
    learn.fine_tune(4, 1e-2)
    lr_new = learn.lr_find().valley
    learn.fine_tune(10, lr_new)

    interp = ClassificationInterpretation.from_learner(learn)
    interp.most_confused(min_val=3)
    interp.plot_top_losses(10, figsize=(15,15))
    interp.plot_top_losses(10, figsize=(15,15))

    return learn


# customize to your code
data_path = Path('/content/CLab21/datasets/emotions/isear')
train_path = data_path / 'isear-train-modified.csv'
valid_path = data_path / 'isear-val-modified.csv'
test_path = data_path / 'isear-test-modified.csv'

train_df, test_df = get_df(train_path, valid_path, test_path)
model = train(train_df)

'''
you can predict your test data using model.predict (as sklearn.)

e.g. model.predit('When I understood that I was admitted to the University.') will asnwer you

>>> ('joy',
 tensor(5),
 tensor([5.9464e-04, 5.7787e-04, 1.3048e-03, 1.5658e-03, 9.9196e-01,
         2.3159e-03, 1.3603e-03]))


that means you have answer 'joy' corresponding to tensor 5 (among 0-6), with '9.9196e-01' probability (which is 99%)
'''


