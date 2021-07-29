"""classify.py; this is core implementation of this library"""
import importlib
import sys
from sklearn.naive_bayes import MultinomialNB as NB
from .model import *
from .data import *
from .interpret import *
from .classify import *
from pathlib import Path

__all__ = ["Classifier"]

model_dict = {'nb': NB, 'mlp': MLP}
# metric_dict = {'f1score': cls_report, 'confusion_matrix': }

class Classifier(DataBunch):
    """
    1. Get data
    2. Convert text data to vectors (in our case, specific embeddings)
    3. Initialize model
    4. train / (and check train results)
    5. evaluate test results
    6. analyse
    """
    def __init__(self, model_type='mlp', emb_type='tfidf', occ_type='', dim=50, verbose=True):
        # import pudb; pudb.set_trace()
        self._config = {}
        self.model_type = model_type.lower()
        self.emb_type = emb_type
        self.occ_type = occ_type
        self.dim = dim
        self.verbose = verbose
        super().__init__()
    
    def __setattr__(self, k, v):
        if not k.startswith("_"): self._config[k] = v
        super().__setattr__(k,v)


    def __call__(self):
        """
        Mainly (transformed) text data and model are created.
        """
        if self.verbose:
            print(f"\n\n\nCurrent configuration: {self._config}\n")
        self.get_data(self.occ_type)
        self.get_embedding(emb_type=self.emb_type, model_type=self.model_type, dim=self.dim)

        if self.verbose: print("\n\n", "="*40, f"\nx_train : {self.x_train.shape}, y_train: {self.y_train.shape}, x_test: {self.x_test.shape}, y_test: {self.y_test.shape}\n\n")
        self.learner = model_dict[self.model_type](self.x_train.shape[1]) if self.model_type=='mlp' else model_dict[self.model_type]()
        
    def train(self):
        """A function actually executes parameter learning."""
        # import pdb; pdb.set_trace()

        self.learner.fit(X=self.x_train, y=self.y_train)

        if self.verbose:
            # temporal value assign and reshaping tensor..
            # TODO: but this is hard coding. we need not to change it as attribute.. (but can change to one hot in model itself.)
            y_train, train_pred = self.y_train, self.learner.predict(self.x_train)
            if y_train.ndim==2: 
                train_pred, y_train = train_pred.argmax(-1), y_train.argmax(-1)
            print("="*60, f"\nAn evaluation report from TRAIN DATA\n\nModel type: {self.learner.__repr__()}\n\n{cls_report(y_train, train_pred)}", sep="\n")


    def predict(self):
        return self.learner.predict(self.x_test)

    def evaluate(self, y_true, y_pred, **kwargs):
        metrics = {k:v for k, v in kwargs}

        if y_true.ndim==2: y_true = y_true.argmax(-1)
        if y_pred.ndim==2: y_pred = y_pred.argmax(-1)

        print("="*60, f"\nAn evaluation report from TEST DATA\n\n\nModel type: {self.learner.__repr__()}.\n{cls_report(y_true, y_pred)}", sep='\n')
        cm_plot = cm(y_true, y_pred, list(self.label2idx.keys()))