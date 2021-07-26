from tensorflow.keras import Model, layers, metrics, losses
from tensorflow_addons.metrics import F1Score  # install 'tensorflow-addons' if you need
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from lime.lime_text import LimeTextExplainer
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
import numpy as np


#  TODO: I can make _package.py or __init__ and set __all__ = [ everything you need ]

class MlpModelLime:

    def __init__(self):
        self.model = None
        self.train_raw = None

    def get_layers(self, shape_of_input):
        input_layer = layers.Input(shape=shape_of_input, name="input")
        dense_layer_1 = layers.Dense(768, activation="relu", name="dense_layer_1")(input_layer)
        dense_layer_2 = layers.Dense(128, activation='relu', name="dense_layer_2")(dense_layer_1)
        dense_layer_3 = layers.Dense(32, activation='relu', name="dense_layer_3")(dense_layer_2)
        dense_layer_4 = layers.Dense(15, activation='relu', name="dense_layer_4")(dense_layer_3)

        output = layers.Dense(7, activation="softmax", name="output")(dense_layer_4)
        #self.model = Model(inputs=input_layer, outputs=self.outputs, name="mlp_lime_model")
        return Model(inputs=input_layer, outputs=output, name="mlp_model")

    def get_model(self, shape_of_input, lr=0.0001, opt=Adam):
        self.model = self.get_layers(shape_of_input)
        optimizer = opt(learning_rate=lr)
        loss = losses.CategoricalCrossentropy(from_logits=True)  # (Jiwon): why from logits?

        # TODO: find a way other than in-place
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            # TODO: regarding f1score, i need to render variable (not integer) referring last layer shape of model
            metrics=[metrics.Recall(), metrics.Precision(), F1Score(num_classes=7)]
        )
        return self.model

    def fit_model(self, x, y):
        '''actual model fitting happens here, and do not have to return model since the model is reference'''

        params = {'epochs': 1, 'bs': 64}
        es = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=5)
        self.model.fit(x=x, y=y, epochs=params['epochs'], batch_size=params['bs'], validation_split=0.2, callbacks=es)


    def lime_predictor(self, text):
        vectorizer = TfidfVectorizer()
        vectorizer.fit_transform(self.train_raw)
        text_vector = vectorizer.transform(text).toarray()
        prob = self.model.predict(text_vector, batch_size=16, verbose=1)
        return prob

    def lime_exp(self, isear_test_x, train_sen):
        y_pred = []
        idx = 31
        self.train_raw = train_sen
        target_names = ["Anger", "Disgust", "Fear", "Guilt", "Joy", "Sadness", "Shame"]

        #self.tfidf_train = vocabulary
        explainer = LimeTextExplainer(class_names=target_names)

        row = isear_test_x[idx]#.split('\t') # now it works for [text, tense, polarity,direction]

        # for column text only:
        # row = isear_test_x[idx].split('\t') and row[0]

        print("Row: %s" % row)

        exp = explainer.explain_instance(row, self.lime_predictor, num_features=7, top_labels=7)

        #print('Document id: %d' % idx)
        #test_vector = self.tfidf_train.tf_idf(isear_test_x, train=False)[:, :3000]
        #prob = self.model.predict(test_vector, batch_size=16, verbose=1)
        #for pred in prob:
            #y_pred.append(np.argmax(pred, axis=0))
        #print('Predicted class: %s' % target_names[y_pred[idx]])
        #print('True class: %s' % y_vector[idx])

        words = exp.as_list()
        exp.save_to_file(f"results_Lime/ISEAR_{idx}.html")

        return exp


