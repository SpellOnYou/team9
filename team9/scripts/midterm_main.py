# -*- coding: utf-8 -*-
# Author: Jiwon Kim, Lara Grimminger

from .utils.loss import CrossEntropy
from .utils.linear import Linear
from .utils.relu import Relu

class Model:
    """This class assigns layers (and its parameters) to layers.
    Parameters
    ----------
    n_layers : {int}, default = 2
        Number of layers in neural networks. Dimension of each parameter is pre-defined.

    args : {listl}
        List of arguments fed to linear layers

    Note
    ----------

    """
    def __init__(self, n_layers, *args):
        
        # make sure our given parameters suffice number of layers
        assert n_layers == len(args)/2, f"The number of arguments and layer should be matched!, number of layers: {n_layers}, number of args: {len(args)/2}"

        self.loss = CrossEntropy()
        self.args = args
        self.layers = []

        #assign layers to be used in forward/backward process
        for i in range(0, n_layers-1):
            self.layers += [Linear(args[i*2], args[(i*2)+1]), Relu()]
        self.layers += [Linear(args[-2], args[-1])]
    
    def forward(self, x):
        """This function is for forward propagation.
        Using defined layers, it recursively feed results of previous layer.
        """
        self.x= x

        # feed forward
        for layer in self.layers:
            x = layer.forward(x)
        self.out = x
        return self.out

    def backward(self):
        """This function is for backward propagation, where derivatives are calculated.
        As in forward processing, it recursively feed results in reverse.
        """

        # Since loss's input was results of last layer, its gradient should be calculated for feeding gradient backward.

        self.loss.backward()
        for layer in reversed(self.layers):
            layer.backward()