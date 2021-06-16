# -*- coding: utf-8 -*-
# Author: Jiwon Kim

class CrossEntropy():
    """Make cross-entropy loss function used for nominal variables.

    Parameters
    ----------
    pred : {torch.tensor}, default = None
        Gets predicion and calculate cross-entropy when instance is called.
        This value is output of last linear layer.

    y : {torch.tensor}, default = None
        Gets target data and calculate cross-entropy.
    
    Note
    ----------
    Here we implemented cross entropy: negative log likelihood of softmax, without splitting calculation of logarithm and softmax.

    """
    def __call__(self, pred=None, y=None):
        # ensure size of data
        assert pred.shape == y.shape, f"Both prediction and y should be in same shape. We get prediction: {pred.shape}, target data: {y.shape}"
        self.yhat, self.y = pred, y

        # calculate log(softmax(x))
        self.log_p_yhat = self.log_softmax(pred)

        # calculate nll(log(softmax(x)))
        self.out = self.nll(self.log_p_yhat, y)

        return self.out

    #negative log likelihood
    def nll(self, pred, y):
        # calculate -sum(x * log p(x))
        # for more information regarding indexing, see: https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.indexing.html#integer-array-indexing
        return -pred[range(y.shape[0]), y.max(-1).indices].mean()

    def log_softmax(self, x): return x - x.exp().sum(-1,keepdim=True).log()

    def backward(self):
        softmax = 1/ (1+(-self.yhat).exp())
        
        # dy/dx = softmax(x) - y, when y = nll(log(softmax(x)))
        self.yhat.g = (softmax - self.y)

class Mse():
    """Mean squared error for ratio scale.
    """
    def __call__(self, pred, y):
        self.yhat, self.y = pred, y
        self.out = (pred-y).pow(2).mean()

    def backward(self):
        self.yhat.g = 2. * (self.yhat - self.y) / self.y.shape[0]

