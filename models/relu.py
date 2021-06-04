# -*- coding: utf-8 -*-
# Author: Jiwon Kim


class Relu():
    """Make Relu(Rectified Linear Unit, https://www.cs.toronto.edu/~fritz/absps/reluICML.pdf) function.

    Attributes
    ----------
    inp.g : tensor
        gradient for input parameter.

    Note
    ----------
    g(gradient) attribute for each paramter doesn't use PyTorch's automatic differentiation engine.
    So you can't access that attribute before backpropagation.

    Examples
    ----------
    x = torch.randn(1000, 10)
    relu = Relu()
    activation = relu(x)
    activation.backward()
    relu_grad = activation.g
    """    
    def forward(self, x):
        """Rectified linear unit activation function.

        Note
        ----------
        The function returns actual return - 0.5 for keeping mean value to near 0.
        """
        self.inp = x
        self.out = x.clamp_min(0.) - 0.5
        return self.out

    def backward(self):
        """
        Note
        ----------
        Since input(i.e. self.inp) is tensor in pytorch, `self.inp>0` is element-wise calculation keeping original dimension, which is rather different from python's list type.
        To make element-wize multiplication possible, we tranformed boolean value to float(so True ->1, False ->0).
        """
        self.inp.g = self.out.g* (self.inp>0).float()