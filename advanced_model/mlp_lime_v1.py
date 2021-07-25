import tensorflow as tf
from tensorflow.keras import Model, layers, metrics, losses
from tensorflow.keras import Model
from F1Score import F1Score
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam, SGD
from lime.lime_text import LimeTextExplainer
from sklearn.feature_extraction.text import TfidfVectorizer

import numpy as np



class LimeExplainer:

    def __init__(self, shape_of_input, epochs=1, bs=64, lr=0.0001, opt=Adam):
        self.input_layer = layers.Input(shape=shape_of_input, name="input")
        self.dense_layer_1 = layers.Dense(768, activation="relu", name="dense_layer_1")(self.input_layer)
        self.dense_layer_2 = layers.Dense(128, activation='relu', name="dense_layer_2")(self.dense_layer_1)
        self.dense_layer_3 = layers.Dense(32, activation='relu', name="dense_layer_3")(self.dense_layer_2)
        self.dense_layer_4 = layers.Dense(15, activation='relu', name="dense_layer_4")(self.dense_layer_3)
        self.outputs = layers.Dense(7, activation="softmax", name="output")(self.dense_layer_4)

        self.epochs = epochs
        self.bs = bs
        self.lr = lr
        self.opt = opt
        self.model = Model(inputs=self.input_layer, outputs=self.outputs, name="advanced_classifier")
        self.tfidf_train = None


    def define_model(self):

        # Define the model
        self.model.summary()
        return self.model

    def fit_model(self, x_train, y_train):

        # Set an optimizer and loss function
        optimizer = self.opt(learning_rate=self.lr)
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

        # Compile the model
        self.model.compile(optimizer=optimizer, loss=loss, metrics=[tf.keras.metrics.Recall(),
                                                                   tf.keras.metrics.Precision(),
                                                                   F1Score()])


        es = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=5)

        history = self.model.fit(x_train,
                                y_train,
                                epochs=self.epochs,
                                batch_size=self.bs,
                                validation_split = 0.2,
                                callbacks=es)
        return history

    # def evaluate(self, test_x, test_y):
    #     y_pred = []
    #     gold_labels = []
    #
    #     self.test_x = test_x
    #
    #     for y in test_y:
    #         gold_labels.append(np.argmax(y))
    #
    #     print("Evaluate")
    #     prediction = self.model.predict(test_x, batch_size=16, verbose=1)
    #     print(type(test_x))
    #     for predicted in prediction:
    #         y_pred.append(np.argmax(predicted, axis=0))
    #
    #     target_names = ["Joy", "Fear", "Shame", "Disgust", "Guilt", "Anger", "Sadness"]
    #     results = classification_report(gold_labels, y_pred, target_names=target_names, output_dict=True, digits=2)
    #     print(results)
    #     df = pd.DataFrame(results).transpose()
    #     df = df.round(2)
    #
    #     self.eval_path = Path('advanced_model/eval_results') / '6layer'
    #     self.eval_path.mkdir(parents=1, exist_ok=1)
    #     opt_name = 'Adam' if self.opt == Adam else 'SGD'
    #     fname = self.eval_path / f"results_{opt_name}_{str(self.bs)}_{str(self.lr)}.csv"
    #     df.to_csv(fname)

    # def lime_predictor(self, text):
    #
    #     text_vector = self.tfidf_train.tf_idf(text, train=False)[:, :3000]
    #     prob = self.model.predict(text_vector, batch_size=16, verbose=1)
    #     return prob

    def lime_predictor(self, text):
        vectorizer = TfidfVectorizer()
        text_vector = vectorizer.fit_transform(text)
        prob = self.model.predict(text_vector, batch_size=16, verbose=1)
        return prob

    def lime_exp(self, isear_test_x):
        y_pred = []
        idx = 31
        target_names = ["Joy", "Fear", "Shame", "Disgust", "Guilt", "Anger", "Sadness"]

        #self.tfidf_train = vocabulary
        explainer = LimeTextExplainer(class_names=target_names)

        row = isear_test_x[idx]
        print("Row: %s" % row)
        
        exp = explainer.explain_instance(row, self.lime_predictor, num_features=6, top_labels=7)

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


