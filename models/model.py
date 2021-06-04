# -*- coding: utf-8 -*-
# Author: Jiwon Kim

class Model():
    def __init__(self, n_layers, *args):
        self.loss = CrossEntropy()
        self.args = args
        self.layers = []
        for i in range(0, n_layers-1):
            self.layers += [Linear(args[i*2], args[(i*2)+1]), Relu()]
        self.layers += [Linear(args[-2], args[-1])]
    
    def forward(self, x):
        self.x= x
        for layer in self.layers:
            x = layer.forward(x)
        self.out = x
        return self.out

    def backward(self):
        self.loss.backward()
        for layer in reversed(self.layers):
            layer.backward()