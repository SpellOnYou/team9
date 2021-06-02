class CrossEntropy():
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