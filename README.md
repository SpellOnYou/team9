
# CLab 21: Group 9 Emotion Classification

Repository for Jiwon Kim and Lara Grimminger for the 2021 Team Laboratory.
Subject: Emotion Classification on the ISEAR Dataset.

## Introduction

Emotion classificaton is the task of classifying the correct emotion given a text.
Here, we have used the ISEAR dataset which contains seven emotion labels (joy, fear, shame, disgust, guilt, anger and sadness) and was obtained by asking students to describe emotional events for 7 emotions.

As baseline, we opted for a simple 2 layer neural network to obtain an understanding of the task
and the way neural networks work.

## Data

You can find the labeled ISEAR dataset in the [datasets/emotions](https://github.com/SpellOnYou/CLab21/tree/main/datasets/emotions) directory. The dataset is split into train, validation and test set. Since the respective datasets contained noise and "not provided" text sequences, we have cleaned the datasets and saved them and added "modified" to the respective file names.

## Input and Output Representation

You can find the code for the input and output representation in the [data_preprocessing](https://github.com/SpellOnYou/CLab21/tree/main/data_preprocessing) directory.
We have used tf-idf to convert the text input to numerical input.
We have used one hot encoding to concert the output labels to numerical labels.

## Model Architecture

You can find the code for the model in the [models](https://github.com/SpellOnYou/CLab21/tree/main/models) directory. We opted for a simple 2 layer neural network with a Relu activation function, a Softmax activation function and Cross Entropy Loss.

The script [linear.py](https://github.com/SpellOnYou/CLab21/blob/main/models/linear.py) contains the linear calculations for the neural network.
The script [loss.py](https://github.com/SpellOnYou/CLab21/blob/main/models/loss.py) contains the calculations for softmax and the Cross Entropy loss for the neural network.
The script [model.py](https://github.com/SpellOnYou/CLab21/blob/main/models/model.py) contains the skeleton code for the neural network.
The script [relu.py](https://github.com/SpellOnYou/CLab21/blob/main/models/relu.py) contains the Relu calculations for the neural netwrok.

The main method of our code is the script with the name [multilayer_perceptron.py](https://github.com/SpellOnYou/CLab21/blob/main/multilayer_perceptron.py)

## Evaluation

You can find the code which contains the metrics to evaluate our model in the [metrics](https://github.com/SpellOnYou/CLab21/tree/main/metrics) directory.
We used Precision, Recall and F1 score.

## Experiment

You can find the results of hyperparameter tuning in the [exp](https://github.com/SpellOnYou/CLab21/tree/main/exp) directory.
We specifically tracked parameters' mean and standard deviation as well as f-score.

Currently we have tested follwing parameter combinations.

Architecture: `2`, `4` and `6-layers` architecture

Max length of features: `100`, `3000`

Batch size: `16`,`32`,`64`

Learning rate: `0.1`, `0.0875`, `0.075`, `0.0625`, `0.05`, `0.0375`, `0.0250`, `0.0125`, `1.00e-10`

Epochs: `1`, `3`, `5`, `7`, `9`, `11`, `13`

Further, we tracked the distribution(i.e. mean, std) of each layer and saved the results as pickles.

## Requirements

numpy 1.19.5


## Tested with

Python 3.8
