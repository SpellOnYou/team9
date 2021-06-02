class Fscore():
    def __init__(self, inp, trg):
        self.inp, self.trg = inp.max(-1).indices, trg.max(-1).indices
        self.c = inp.shape[1]
    def __call__(self, alpha = 0.5):
        self.precision()
        self.recall()
        f1 = map(
            self.fscore, self.tot_pre, self.tot_rec
            )
        return self.tot_pre, self.tot_rec, list(f1)
    def fscore(self, x, y):
        return (2*x*y)/(x+y)
    
    def precision(self):
        self.tot_pre= []
        for i in range(self.c):
            numer = self.inp == self.trg
            denom = self.inp ==i
            if not sum(denom)==0: self.tot_pre += [sum(numer) / sum(denom)]
            else: self.tot_pre += [0.]

    def recall(self):
        self.tot_rec= []
        for i in range(self.c):
            numer = self.inp == self.trg
            denom = self.trg ==i
            if not sum(denom)==0: self.tot_rec += [sum(numer) / sum(denom)]
            else: self.tot_rec += [0.]