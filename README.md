# CLab 21: Group 9 Emotion Classification


### Overview



### Installation

Install this project from the source by 
```sh
$ git clone https://github.com/SpellOnYou/nlp-tema9
$ cd nlp-team9
$ pip install .
```


### Usage

#### Python Module (GUI)

On a high level, we provide a python object `team9.Classifier` and the supported features like: loading data, vectorizing/embedding text, create model, train dataset, predict from trained model, and analysing the results.
Mainly, this module consists of several submodules: [data]() where loading and embedding happen, [model]() which executes actual training and prediction, [interpret] which provides relevant metrics to evaluate model estimation, and finally [classify]() where integrate the submodules and renders various options to the submodules repectively.

Here is the most succinct version of example, 

```python
import team9
clf = team9.Classifier(model_type='MLP',emb_type='tfidf', occ_type='rule')
clf.train()
pred = clf.predict()
clf.valiate(clf.y_test, pred)

```

and please refer to our source code for implement details and try the demo on [Google Colab](https://colab.research.google.com/drive/1eWcxVjaEadUxoMwy9GCJ9_N9-67ussKC?usp=sharing)


#### Command Line Intergface (CLI)

We provide a command line interface (CLI) of emotion classification (of ISEAR dataset, which can be easily extended to other datasets) as well as the python module.

As for additional available arguments, please see 

```sh
team9-emo-cls -h
```

----- this version was written for final -----
TODO: This should be intergrated with below 👇 description
---


Repository for [Jiwon Kim](mailto:st176776@stud.uni-stuttgart.de) and Lara Grimminger for the 2021 Team Laboratory.
Subject: Emotion Classification on the ISEAR Dataset.

## Introduction

Emotion classificaton is the task of classifying the correct emotion given a text.
Here, we have used the ISEAR dataset which contains seven emotion labels (joy, fear, shame, disgust, guilt, anger and sadness) and was obtained by asking students to describe emotional events for 7 emotions.

As baseline, we opted for a simple 2 layer neural network to obtain an understanding of the task
and the way neural networks work.

### Data

You can find the labeled ISEAR dataset in the [datasets/emotions](https://github.com/SpellOnYou/CLab21/tree/main/datasets/emotions) directory. The dataset is split into train, validation and test set. Since the respective datasets contained noise and "not provided" text sequences, we have cleaned the datasets and saved them and added "modified" to the respective file names.

### Input and Output Representation

You can find the code for the input and output representation in the [data_preprocessing](https://github.com/SpellOnYou/CLab21/tree/main/data_preprocessing) directory.
We have used tf-idf to convert the text input to numerical input.
We have used one hot encoding to concert the output labels to numerical labels.

### Model Architecture

You can find the code for the model in the [models](https://github.com/SpellOnYou/CLab21/tree/main/models) directory. We opted for a simple 2 layer neural network with a Relu activation function, a Softmax activation function and Cross Entropy Loss.

The script [linear.py](https://github.com/SpellOnYou/CLab21/blob/main/models/linear.py) contains the linear calculations for the neural network.
The script [loss.py](https://github.com/SpellOnYou/CLab21/blob/main/models/loss.py) contains the calculations for softmax and the Cross Entropy loss for the neural network.
The script [model.py](https://github.com/SpellOnYou/CLab21/blob/main/models/model.py) contains the skeleton code for the neural network.
The script [relu.py](https://github.com/SpellOnYou/CLab21/blob/main/models/relu.py) contains the Relu calculations for the neural netwrok.

The main method of our code is the script with the name [multilayer_perceptron.py](https://github.com/SpellOnYou/CLab21/blob/main/multilayer_perceptron.py)

### Evaluation

You can find the code which contains the metrics to evaluate our model in the [metrics](https://github.com/SpellOnYou/CLab21/tree/main/metrics) directory.
We used Precision, Recall and F1 score.


## Usage

To use experiment with your MLP, we recommend you to clone this repository to your local and test in your command line interface(CLI) as following:

```git clone https://github.com/SpellOnYou/CLab21.git <your_repo_name>
cd <your_repo_name>
python multilayer_perceptron.py```

or if you want to track(default:False) the learning status or tune the number of layer size(default:2), simply change the command to

```
python multilayer_perceptron.py trace=True n_layers=4
```

Also, you can find the results of various hyperparameter we've already experimented in the [exp](https://github.com/SpellOnYou/CLab21/tree/main/exp) directory.
We specifically tracked parameters' mean and standard deviation as well as f-score.

Currently we have tested follwing parameter combinations.

Architecture: `2`, `4` and `6-layers` architecture

Max length of features: `100`, `3000`

Batch size: `16`,`32`,`64`

Learning rate: `0.1`, `0.0875`, `0.075`, `0.0625`, `0.05`, `0.0375`, `0.0250`, `0.0125`, `1.00e-10`

Epochs: `1`, `3`, `5`, `7`, `9`, `11`, `13`

## Requirements

numpy version >= 1.19
python version >= 3.6
pytorch version >= 1.0
