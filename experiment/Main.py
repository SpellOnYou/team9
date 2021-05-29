import numpy as np
import csv, math
from torch import tensor, float32, randn, no_grad
from BOW import PadMaxLength, BagOfWords
from OneHotEncoding import OneHotEncoding
from DummyModel import DummyModel
from F1Score import Fscore
from sklearn.metrics import classification_report
import pandas as pd

train_file =  "/home/lara/PycharmProjects/pythonProject/isear-train-modified.csv"
val_file =  "/home/lara/PycharmProjects/pythonProject/isear-val-modified.csv"
test_file =  "/home/lara/PycharmProjects/pythonProject/isear-test-modified.csv"

pml_train = PadMaxLength(train_file)
pml_val = PadMaxLength(val_file)
pml_test = PadMaxLength(test_file)

max_sent, min_sent = pml_train.min_max_sentences()

sentences_padded_train = pml_train.right_pad_sentences(max_sent)
sentences_padded_val = pml_val.right_pad_sentences(max_sent)
sentences_padded_test = pml_test.right_pad_sentences(max_sent)


vocab_list = pml_train.merge_with(sentences_padded_val, sentences_padded_test)  # Vocab over all files

bow_train = BagOfWords(vocab_list, sentences_padded_train)  # Sentences to create the vocabulary
bow_val = BagOfWords(vocab_list, sentences_padded_val)
bow_test = BagOfWords(vocab_list, sentences_padded_test)

tf_idf_train = bow_train.tf_idf(sentences_padded_train)
tf_idf_val = bow_val.tf_idf(sentences_padded_val)
tf_idf_test = bow_test.tf_idf(sentences_padded_test)


ohe_train = OneHotEncoding(train_file)
ohe_val = OneHotEncoding(val_file)
ohe_test = OneHotEncoding(test_file)

y_train = ohe_train.one_hot_encoding()
reference_dict = ohe_train.get_encoded_dict()

y_val = ohe_val.one_hot_encoding(reference_dict)
y_test = ohe_test.one_hot_encoding(reference_dict)

train_x, train_y = tensor(tf_idf_train, dtype=float32), tensor(y_train)
valid_x, valid_y = tensor(tf_idf_val, dtype=float32), tensor(y_val)

print(train_x.shape, valid_x.shape)

train_x = train_x[:, :3000]
valid_x = valid_x[:, :3000]

train_x = train_x
valid_x = valid_x

n, m, l, h, o, c = *train_x.shape, 145, 32, 15, train_y.shape[1]
#n, m, h, c = *train_x.shape, 100, train_y.shape[1]
print(train_x.shape)
print(*train_x.shape)
print(train_y.shape[1])
#
# w1 = randn(m, h) / math.sqrt(h)
# w2 = randn(h, c)
# b1 = randn(h)
# b2 = randn(c)

w1 = randn(m, l) / math.sqrt(h) # 3000, 145
w2 = randn(l, h) # 145, 32
w3 = randn(h, o) # 32, 15
w4 = randn(o, c) # 15, 7
b1 = randn(l) # 145
b2 = randn(h) # 32
b3 = randn(o) # 15
b4 = randn(c) # 7

#model = DummyModel(w1, b1, w2, b2)
model = DummyModel(w1, b1, w2, b2, w3, b3, w4, b4)

def train(epochs, bs, lr):
    for e in range(epochs):
        for bs_i in range((n - 1) // bs + 1):
            tot_w_mean, tot_w_std = 0, 0
            str_idx, end_idx = bs_i * bs, (bs_i + 1) * bs
            x_batch, y_batch = train_x[str_idx:end_idx], train_y[str_idx:end_idx]
            prediction = model.forward(x_batch)
            loss = model.loss(prediction, y_batch)

            model.backward(model.layers[-1].inp)

            with no_grad():
                for layer in model.layers:
                    if hasattr(layer, 'w'):  # if they have parameter attribute
                        tot_w_mean += layer.w.g.mean()
                        tot_w_std += layer.w.g.std()
                        layer.w -= layer.w.g * lr
                        layer.b -= layer.b.g * lr
                        layer.w.g.zero_()  # initialize them to zero
                        layer.b.g.zero_()


def evaluate(epoch_nbr, learning_rate):
    train(epoch_nbr, 32, learning_rate)

    # loss after training
    pred = model.forward(train_x[:32])
    loss = model.loss(pred, train_y[:32])

    # Evaluation
    pred_valid = model.forward(valid_x)
    loss_valid = model.loss(pred_valid, valid_y)
    softmax_pred = model.loss.log_softmax(model.loss.yhat)

    measure = Fscore(softmax_pred, valid_y)
    p, r, f = measure()

    trg_names = list(reference_dict.keys())
    report = (classification_report(y_true = measure.trg, y_pred = measure.inp, target_names=trg_names, output_dict=True))
    df = pd.DataFrame(report).transpose()
    df = df.round(2)
    results = df.to_csv("results_epoch_" + str(epoch_nbr) + "_rate_" + str(learning_rate) + ".csv")

epochs = [5, 10, 20]
lr_nbr = [0.01, 0.001, 0.0001, 0.00001]

for e in epochs:
    for lr in lr_nbr:
        evaluate(e, lr)



