import tensorflow as tf
from tensorflow.keras import Model
from F1Score import F1Score
from sklearn.metrics import classification_report
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam, SGD
import pandas as pd
from sklearn.metrics import confusion_matrix



class MultipleInputsModel:

    def __init__(self, len_of_features, len_of_input, epochs=20, bs=64, lr=0.0001, opt=Adam):
        self.inputs_1 = tf.keras.layers.Input(shape=len_of_input, name="input_1")
        self.inputs_2 = tf.keras.layers.Input(shape=(len_of_features,), name="input_2")

        self.model_layer = tf.keras.layers.Dense(512, activation="relu", name="model_layer")(self.inputs_1)

        self.dense_layer_1 = tf.keras.layers.Dense(512, activation="relu", name="dense_layer_1")(self.inputs_2)
        self.dense_layer_2 = tf.keras.layers.Dense(256, activation="relu", name="dense_layer_2")(self.dense_layer_1)

        self.concat_layer = Concatenate()([self.model_layer, self.dense_layer_2])

        self.dense_layer_3 = tf.keras.layers.Dense(128, activation='relu', name="dense_layer_3")(self.concat_layer)
        self.dense_layer_4 = tf.keras.layers.Dense(32, activation='relu', name="dense_layer_4")(self.dense_layer_3)
        self.dense_layer_5 = tf.keras.layers.Dense(15, activation='relu', name="dense_layer_5")(self.dense_layer_4)
        self.outputs = tf.keras.layers.Dense(7, activation="softmax", name="output")(self.dense_layer_5)

        self.epochs = epochs
        self.bs = bs
        self.lr = lr
        self.opt = opt

    def define_model(self):

        # Define the model
        model = Model(inputs=[self.inputs_1, self.inputs_2], outputs=self.outputs, name="multiple_inputs_model")
        model.summary()
        return model

    def fit_model(self, dnn_model, x1_train, x2_train, y_train):

        # Set an optimizer and loss function
        optimizer = self.opt(learning_rate=self.lr)
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

        # Compile the model
        dnn_model.compile(optimizer=optimizer, loss=loss, metrics=[tf.keras.metrics.Recall(),
                                                                   tf.keras.metrics.Precision(),
                                                                   F1Score()])

        es = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=5)

        # Fit model on training data
        history = dnn_model.fit(x=[x1_train, x2_train],
                                y=y_train,
                                epochs=self.epochs,
                                batch_size=self.bs,
                                validation_split=0.2,
                                callbacks=es)
        return history

    def evaluate(self, dnn_model, test1_x, test2_x, test_y):
        y_pred = []
        gold_labels = []

        for y in test_y:
            gold_labels.append(np.argmax(y))

        print("Evaluate")
        prediction = dnn_model.predict(x=[test1_x, test2_x], batch_size=64, verbose=1)

        for predicted in prediction:
            y_pred.append(np.argmax(predicted, axis=0))

        target_names = ["Joy", "Fear", "Shame", "Disgust", "Guilt", "Anger", "Sadness"]
        results = classification_report(gold_labels, y_pred, target_names=target_names)
        results_to_csv = classification_report(gold_labels, y_pred, target_names=target_names, output_dict=True)
        df_to_csv = pd.DataFrame(results_to_csv).transpose()
        df = df_to_csv.round(2)

        cmtx = pd.DataFrame(
            confusion_matrix(gold_labels, y_pred, labels=[0, 1, 2, 3, 4, 5, 6]),
            index=['true:0', 'true:1', 'true:2', 'true:3', 'true:4', 'true:5', 'true:6'],
            columns=['pred:0', 'pred:1', 'pred:2', 'pred:3', 'pred:4', 'pred:5', 'pred:6'])

        return results, cmtx
