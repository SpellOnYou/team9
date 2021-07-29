"""mlp.py"""
import sys

import tensorflow as tf
from tensorflow_addons.metrics import F1Score
from functools import partial

hparams = {
    'learning_rate': 1e-4,
    'optimizer': tf.keras.optimizers.Adam,    
    'loss': tf.keras.losses.CategoricalCrossentropy(),
    'metrics': [tf.keras.metrics.Recall(), tf.keras.metrics.Precision(), F1Score(num_classes=7)]
}

params = {
    'epoch': 1,
    'batch_size': 64,
    'validation_split': 0.2,
    'callbacks': tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=5)
}

class MLP():
    """
    Fully-connected multi-layer perceptron. Note here we fix the length of layers.
    """
    def __init__(self, *args, **kwargs):
        """
        A function configure hyper parameters, change it if in need
        (Hyper)parameters are previously set up (as others), but can be changed when user gives adequage values.

        Method
        ---
        _config_hparams
            This is private function, and used for change hyperparameters by checking given arguments
        """
        
        self.hparams = self._config_params(hparams, **kwargs)
        #Set model
        opt = self.hparams['optimizer']
        self.model = self._get_layers(*args)
        self.model.compile(
            optimizer = opt(learning_rate = self.hparams['learning_rate']),
            loss = self.hparams['loss'],
            metrics = self.hparams['metrics']
        )

    def _config_params(self, trg_params, **kwargs):
        """
        Change hyperparameters or params when given name and type are adequate
        Parameter
        --- 
            (default) parameters: (dict)
        Return
        ---
            (un)modified parameters: (dict)
        """
        kwargs = {k:v for k, v in kwargs}
        for k, v in kwargs.items():
            if k in trg_params and isinstance(v, type(kwargs[k])):
                if kwargs['verbose']: sys.out('')
                trg_params[k] = v
        return trg_params

    def _get_layers(self, *args):
        self.input_layer = tf.keras.layers.Input(shape=args[0], name="input_1")
        self.dense_layer_1 = tf.keras.layers.Dense(512, activation="relu", name="dense_layer_1")(self.input_layer)
        self.dense_layer_2 = tf.keras.layers.Dense(256, activation="relu", name="dense_layer_2")(self.dense_layer_1)
        self.dense_layer_3 = tf.keras.layers.Dense(128, activation='relu', name="dense_layer_3")(self.dense_layer_2)
        self.dense_layer_4 = tf.keras.layers.Dense(32, activation='relu', name="dense_layer_4")(self.dense_layer_3)
        self.dense_layer_5 = tf.keras.layers.Dense(15, activation='relu', name="dense_layer_5")(self.dense_layer_4)
        self.outputs = tf.keras.layers.Dense(7, activation="softmax", name="output")(self.dense_layer_5)
        
        return tf.keras.Model(inputs=self.input_layer, outputs=self.outputs, name="multiple_inputs_model")
        
    def fit(self, X, y, **kwargs):

        print(X.shape, y.shape)
        self.model.summary()
        
        self.params = self._config_params(params, **kwargs)
        # import pudb; pudb.set_trace()
        self.model.fit(
            x=X,
            y=y,
            epochs = self.params['epoch'],
            batch_size = self.params['batch_size'],
            validation_split=self.params['validation_split'],
            callbacks=self.params['callbacks'])
       
    def predict(self, x):
        return self.model.predict(x)