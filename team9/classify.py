"""classify.py; this is core implementation of this library"""
import importlib

from sklearn.naive_bayes import MultinomialNB as NB
from .model import *
from .data import *
from .interpret import *
from .classify import *
from pathlib import Path

__all__ = ["Classifier"]

model_dict = {'nb': NB, 'mlp': MLP}

class Classifier(DataBunch):
    """
    1. Get data
    2. Convert text data to vectors (in our case, specific embeddings)
    3. Initialize model
    4. train / (and check train results)
    5. evaluate test results
    6. analyse
    """
	def __init__(self, *args, **kwargs):
		super().__init__()
		self.kwargs = {k: v for k, v in kwargs.items()}
		self.model = model_dict[self.kwargs['model_type'].lower()]		

	def train(self, x, y):

		return train_pred

	def __call__(self):
        """
        Mainly (transformed) text data and model are created.
        """
		self.get_data()
		self.get_embedding()
        self.model()

    def train(self):
        """A function actually executes parameter learning."""
		self.model.fit(self.x_train, self.y_train)

		if 'verbose' in self.kwargs:
			train_pred = self.model.predict(self.x_train)
			print(f"Model type: {self.model.__repr__()}. An evaluation report from train data\n{cls_report(self.y_train, train_pred)}")                
    
    def evaluate(self):
		test_pred = self.model.predict(self.x_test)

		print(f"Model type: {self.model.__repr__()}. An evaluation report from test data\n{cls_report(self.y_test, test_pred)}")
