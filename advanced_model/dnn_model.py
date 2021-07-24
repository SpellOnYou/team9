import tensorflow as tf
from tensorflow.keras import Model
from F1Score import F1Score
from sklearn.metrics import classification_report
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam, SGD
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
from yellowbrick.classifier import ConfusionMatrix


class DnnModel:

    def __init__(self, epochs=20, bs=64, lr=0.0001, opt=Adam):
        self.inputs = tf.keras.layers.Input(shape=(3000,))
        #self.inputs = tf.keras.layers.Input(shape=(2,))
        self.l1 = tf.keras.layers.Dense(512, activation="relu")(self.inputs)
        self.l2 = tf.keras.layers.Dense(256, activation="relu")(self.l1)
        self.l3 = tf.keras.layers.Dense(128, activation="relu")(self.l2)
        self.l4 = tf.keras.layers.Dense(32, activation="relu")(self.l3)
        self.l5 = tf.keras.layers.Dense(15, activation="relu")(self.l4)
        self.outputs = tf.keras.layers.Dense(7, activation="softmax")(self.l3)

        self.epochs = epochs
        self.bs = bs
        self.lr = lr
        self.opt = opt

    def define_model(self):

        # Define the model
        model = Model(inputs=self.inputs, outputs=self.outputs, name="advanced_classifier")
        model.summary()
        return model

    def fit_model(self, dnn_model, x_train, y_train):

        # Set an optimizer and loss function
        optimizer = self.opt(learning_rate=self.lr)
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

        # Compile the model
        dnn_model.compile(optimizer=optimizer, loss=loss, metrics=[tf.keras.metrics.Recall(),
                                                                   tf.keras.metrics.Precision(),
                                                                   F1Score()])

        es = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=5)

        # Fit model on training data
        history = dnn_model.fit(x_train,
                                y_train,
                                epochs=self.epochs,
                                batch_size=self.bs,
                                validation_split=0.2,
                                callbacks=es)
        return history

    def evaluate(self, dnn_model, test_x, test_y):
        y_pred = []
        gold_labels = []

        for y in test_y:
            gold_labels.append(np.argmax(y))

        prediction = dnn_model.predict(test_x, batch_size=64, verbose=1)

        for predicted in prediction:
            print(predicted)
            y_pred.append(np.argmax(predicted, axis=0))

        target_names = ["Joy", "Fear", "Shame", "Disgust", "Guilt", "Anger", "Sadness"]
        results = classification_report(gold_labels, y_pred, target_names=target_names)
        #results_to_csv = classification_report(gold_labels, y_pred, target_names=target_names, output_dict=True)
        #df_to_csv = pd.DataFrame(results_to_csv).transpose()
        #df = df_to_csv.round(2)

        cmtx = pd.DataFrame(
            confusion_matrix(gold_labels, y_pred, labels=[0, 1, 2, 3, 4, 5, 6]),
            index=['true:0', 'true:1', 'true:2', 'true:3', 'true:4', 'true:5', 'true:6'],
            columns=['pred:0', 'pred:1', 'pred:2', 'pred:3', 'pred:4', 'pred:5', 'pred:6'])

        return results, cmtx


    def analyse(self, dnn_model, test_x, test_y, isear_test_x):
        target_names = ["Joy", "Fear", "Shame", "Disgust", "Guilt", "Anger", "Sadness"]

        y_pred = []
        gold_labels = []

        for y in test_y:
            gold_labels.append(np.argmax(y))

        prediction = dnn_model.predict(test_x, batch_size=64, verbose=1)

        for predicted in prediction:
            y_pred.append(np.argmax(predicted, axis=0))

        idx = 0
        for prediction, label in zip(y_pred, gold_labels):
            if prediction != label:
                print("Id: %s" % isear_test_x.index(isear_test_x[idx]))
                print("Sample:  %s" % isear_test_x[idx])
                print("has been classified as", target_names[prediction])
                print("and should be", target_names[label])
                print("###############################################################################################")
            if prediction == label:
                print("Id: ", isear_test_x.index(isear_test_x[idx]),
                      "Sample: ", isear_test_x[idx],
                      ", has been classified as",
                      target_names[prediction],
                      "and is", target_names[label])
                print(
                    "################################################################################################")
            idx += 1
