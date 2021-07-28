"""mlp.py"""
import sys

import tensorflow as tf
from tensorflow_addons.metrics import F1Score

hparams = {
    'learning_rate': 1e-4,
    'optimizer': Adam,
    'loss': tf.keras.losses.CategoricalCrossEntropy(from_logits=True),
    'metrics': [tf.keras.metrics.Recall, tf.keras.metrics.Precision, F1Score]
}

params = {
    'epoch': 20,
    'batch_size': 64,
    'validation_split': 0.2,
    'callbacks': tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=5)
}

class MLP():
    """
    Fully-connected multi-layer perceptron. Note here we fix the length of layers.
    """
	def __init__(self):
        """
        A function configure hyper parameters, change it if in need, and create model.
        (Hyper)parameters are previously set up (as others), but can be changed when user gives adequage values.

        Method
        ---
        _config_hparams
            This is private function, and used for change hyperparameters by checking given arguments
        """
        
        self.hparams = self._config_params(hparams)
        
        #Set model
        opt = self.hparams['optimizer']
        self.model = self._get_layers.compile(
            optimizer = opt(learning_rate = self.hparams['learning_rate']),
            loss = self.hparams['loss'],
            metrics = self.hparams['metrics']
        )
        
    def _config_params(self, trg_params):
        """
        Change hyperparameters or params when given name and type are adequate
        Parameter
        --- 
            (default) parameters: (dict)
        Return
        ---
            (un)modified parameters: (dict)
        """
        for k, v in self.kwargs.items():
            if k in trg_params and isinstance(v, type(self.kwargs[k])):
                if self.kwargs['verbose']: sys.out('')
                trg_params[k] = v
        return trg_params

    @property
    def get_layers(self):
        input_layer = tf.keras.layers.Input(shape=self.x_train.shape[1], name="input_1")
        dense_layer_1 = tf.keras.layers.Dense(512, activation="relu", name="dense_layer_1")(input_layer)
        dense_layer_2 = tf.keras.layers.Dense(256, activation="relu", name="dense_layer_2")(dense_layer_1)
        dense_layer_3 = tf.keras.layers.Dense(128, activation='relu', name="dense_layer_3")(dense_layer_2)
        dense_layer_4 = tf.keras.layers.Dense(32, activation='relu', name="dense_layer_4")(dense_layer_3)
        dense_layer_5 = tf.keras.layers.Dense(15, activation='relu', name="dense_layer_5")(dense_layer_4)
        outputs = tf.keras.layers.Dense(y_train.max()+1, activation="softmax", name="output")(dense_layer_5)
        
        model_layers = tf.keras.Model(inputs=self.input_layer, outputs=self.outputs, name="multiple_inputs_model")

        if self.kwargs['verbose']: model.summary()

        return model_layers
        
    def train(self, x, y):
        self.params = self._config_params(params)

        model.fit(
        	x=x,
        	y=y,
        	epochs = self.params['epoch'],
        	batch_size = self.params['batch_size'],
        	validation_split=self.params['validation_split'],
        	callbacks=self.params['callbacks'])
        
        # model.predict()
        
    def predict(self,x):
        return model.predict(x).argmax(-1)