class Fscore():
    """
    This class is to calculate fsocre of our model

    Parameters
    ----------
    inp : {torch.tensor}
        one hot vector prediction from model

    trg : {torch.tensor}
        ground-truth label

    """
    def __init__(self, inp, trg):
        """get prediction and target label, change them to index of label to make calculation easier
        """

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
        """This functin calculates precision, i.e. proportion of |collect answer| and |label in target|
        """
        self.tot_pre= []
        for i in range(self.c):
            
            matched = (self.inp == self.trg)
            whole_num = (self.inp==i)
            
            # check logical and
            numer = matched & whole_num
            # |label in target|
            denom = self.trg ==i

            if not sum(denom)==0: self.tot_pre += [(sum(numer) / sum(denom)).tolist()]
            else: self.tot_pre += [0.]

    def recall(self):
        """This functin calculates precision, i.e. proportion of |collect answer| and |label in prediction|
        """

        self.tot_rec= []
        for i in range(self.c):
            # number of collect answers
            matched = (self.inp == self.trg)
            whole_num = (self.inp==i)
            
            # check logical and
            numer = matched & whole_num

            # |label in prediction|
            denom = self.inp ==i

            if not sum(denom)==0: self.tot_rec += [(sum(numer) / sum(denom)).tolist()]
            else: self.tot_rec += [0.]