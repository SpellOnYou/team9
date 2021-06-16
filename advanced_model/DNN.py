import tensorflow as tf
from tensorflow.keras import Model
from F1Score import F1_Score
from sklearn.metrics import classification_report
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam, SGD
import pandas as pd


class DNN_Model():

    def __init__(self, epochs=10, bs=16, lr=0.025, opt=Adam):
        self.inputs = tf.keras.layers.Input(shape=(3000,))
        self.l1 = tf.keras.layers.Dense(145, activation="relu")(self.inputs)
        self.l2 = tf.keras.layers.Dense(32, activation="relu")(self.l1)
        self.l3 = tf.keras.layers.Dense(15, activation="relu")(self.l2)
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

    def compile_fit_model(self, dnn_model, x_train, y_train, x_val, y_val):

        # Set an optimizer and loss function
        optimizer = self.opt(learning_rate=self.lr)
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

        # Compile the model
        dnn_model.compile(optimizer=optimizer, loss=loss, metrics=[tf.keras.metrics.Recall(),
                                                                   tf.keras.metrics.Precision(),
                                                                   F1_Score()])

        es = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=5)

        # Fit model on training data
        history = dnn_model.fit(x_train,
                                y_train,
                                epochs=self.epochs,
                                batch_size=self.bs,
                                validation_data=(x_val, y_val),
                                callbacks=es)
        return history

    def evaluate(self, dnn_model, test_x, test_y):
        y_pred = []
        gold_labels = []

        for y in test_y:
            gold_labels.append(np.argmax(y))

        print("Evaluate")
        prediction = dnn_model.predict(test_x, batch_size=16, verbose=1)

        for predicted in prediction:
            y_pred.append(np.argmax(predicted, axis=0))

        target_names = ["Joy", "Fear", "Shame", "Disgust", "Guilt", "Anger", "Sadness"]
        results = classification_report(gold_labels, y_pred, target_names=target_names, output_dict=True, digits=2)
        df = pd.DataFrame(results).transpose()
        df = df.round(2)
        if self.opt is Adam:
            df.to_csv("results_Adam_" + str(self.bs) + "_" + str(self.lr) + ".csv")
        else:
            df.to_csv("results_SGD_" + str(self.bs) + "_" + str(self.lr) + ".csv")

        return results

