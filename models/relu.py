class Relu:
    def forward(self, x):
        self.inp = x
        self.out = x.clamp_min(0.) - 0.5
        return self.out

    def backward(self):
        self.inp.g = self.out.g * (self.inp > 0).float()
