class Linear():
    def __init__(self, w, b):
        self.w, self.b = w, b

    def forward(self, x):
        self.inp = x
        self.out = self.inp @ self.w + self.b
        return self.out

    def backward(self):
        # set_trace()
        self.inp.g = self.out.g @ self.w.t()
        self.w.g = (self.inp.unsqueeze(-1) * self.out.g.unsqueeze(1)).sum(0)
        self.b.g = self.out.g.sum(0)


class Relu():
    def forward(self, x):
        self.inp = x
        self.out = x.clamp_min(0.) - 0.5
        return self.out

    def backward(self):
        self.inp.g = self.out.g * (self.inp > 0).float()

class CrossEntropy():
    def __call__(self, pred, y):
        self.yhat, self.y = pred, y
        # P(\hat{y})
        self.log_p_yhat = self.log_softmax(pred)
        self.out = self.nll(self.log_p_yhat, y)

        return self.out

    # negative log likelihood
    def nll(self, pred, y):
        # print(pred.shape, y.shape)
        return -pred[range(y.shape[0]), y.max(-1).indices].mean() # The loss is the mean of the individual NLLs

    def log_softmax(self, x): return x - x.exp().sum(-1, keepdim=True).log()

    def backward(self, inp):
        # set_trace()
        self.yhat.g = (inp.unsqueeze(1) * (self.yhat - self.y).unsqueeze(-1)).sum(-1)

class Mse():
    def __call__(self, yhat, y):
        # set_trace()
        self.yhat, self.y = yhat, y
        self.out = (yhat.squeeze(-1) - y).pow(2).mean()
        return self.out

    def backward(self):
        self.yhat.g = 2. * (self.yhat.squeeze() - self.y).unsqueeze(-1) / self.y.shape[0]


class DummyModel():
    def __init__(self, w1, b1, w2, b2, w3, b3, w4, b4):
        self.loss = CrossEntropy()
        self.layers = [Linear(w1, b1), Relu(), Linear(w2, b2), Relu(), Linear(w3, b3), Relu(), Linear(w4, b4)]

    def forward(self, x):
        self.x = x

        for layer in self.layers:
            x = layer.forward(x)
        self.out = x
        return self.out

    def backward(self, x2):
        self.loss.backward(x2)
        for layer in reversed(self.layers):
            layer.backward()