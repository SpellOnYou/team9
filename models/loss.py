# -*- coding: utf-8 -*-
# Author: Jiwon Kim

class CrossEntropy():
    """Make cross-entropy loss function used for nominal variables.

    Parameters
    ----------
    pred : {torch.tensor}, default = None
        Gets predicion and calculate cross-entropy when instance is called.

    y : {torch.tensor}, default = None
        Gets y(target value) and calculate cross-entropy when instance is called.


    """
    def __call__(self, pred, y):
        
        self.yhat, self.y = pred, y
        #P(\hat{y})
        self.log_p_yhat = self.log_softmax(pred)
        self.out = self.nll(self.log_p_yhat, y)
        
        return self.out

    #negative log likelihood
    def nll(self, pred, y):
        # print(pred.shape, y.shape)
        return -pred[range(y.shape[0]), y.max(-1).indices].mean()

    def log_softmax(self, x): return x - x.exp().sum(-1,keepdim=True).log()

    def backward(self):
        softmax = 1/ (1+(-self.yhat).exp())
        # set_trace()
        self.yhat.g = (softmax - self.y)

class Mse():
    """
    """
    def __call__(self, pred, y):
        self.yhat, self.y = pred, y
        self.out = (pred-y).pow(2).mean()

    def backward(self):
        self.yhat.g = 2. * (self.yhat - self.y) / self.y.shape[0]

