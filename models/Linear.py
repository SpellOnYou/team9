class Linear():
    def __init__(self, w, b):
        self.w, self.b = w, b
    def forward(self, x): 
        self.inp = x
        self.out = self.inp@self.w + self.b
        return self.out
    
    def backward(self):
        # set_trace()
        self.inp.g = self.out.g @ self.w.t()
        self.w.g = (self.inp.unsqueeze(-1) * self.out.g.unsqueeze(1)).sum(0)
        self.b.g = self.out.g.sum(0)