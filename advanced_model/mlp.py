from tensorflow.keras import Model, layers, metrics, losses
from tensorflow_addons.metrics import F1Score  # install 'tensorflow-addons' if you need
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam


#  TODO: I can make _package.py or __init__ and set __all__ = [ everything you need ]

class MlpModel:

    def get_layers(self, shape_of_input):
        input_layer = layers.Input(shape=shape_of_input, name="input")
        dense_layer_1 = layers.Dense(768, activation="relu", name="dense_layer_1")(input_layer)
        dense_layer_2 = layers.Dense(128, activation='relu', name="dense_layer_2")(dense_layer_1)
        dense_layer_3 = layers.Dense(32, activation='relu', name="dense_layer_3")(dense_layer_2)
        dense_layer_4 = layers.Dense(15, activation='relu', name="dense_layer_4")(dense_layer_3)

        output = layers.Dense(7, activation="softmax", name="output")(dense_layer_4)

        return Model(inputs=input_layer, outputs=output, name="mlp_model")

    def get_model(self, shape_of_input, lr=0.0001, opt=Adam):
        model = self.get_layers(shape_of_input)
        optimizer = opt(learning_rate=lr)
        loss = losses.CategoricalCrossentropy(from_logits=True)  # (Jiwon): why from logits?

        # TODO: find a way other than in-place
        model.compile(
            optimizer=optimizer,
            loss=loss,
            # TODO: regarding f1score, i need to render variable (not integer) referring last layer shape of model
            metrics=[metrics.Recall(), metrics.Precision(), F1Score(num_classes=7)]
        )
        return model

    def fit_model(self, model, x, y):
        '''actual model fitting happens here, and do not have to return model since the model is reference'''

        params = {'epochs': 20, 'bs': 64}
        es = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=5)
        model.fit(x=x, y=y, epochs=params['epochs'], batch_size=params['bs'], validation_split=0.2, callbacks=es)
