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
	def __init__(self, model_type = 'MLP', *args, **kwargs):
		# import pudb; pudb.set_trace()
		super().__init__()
		self.model_type = model_type
		self.kwargs = {k: v for k, v in kwargs.items()}
		print(self.kwargs)
		self.get_data()
		self.get_embedding()
		# if self.kwargs['model_type'].lower() == 'mlp': 
		self.model = model_dict[model_type.lower()](self.x_train.shape[1])
		
	def train(self):
		"""A function actually executes parameter learning."""

		self.model.fit(x=self.x_train, y=self.y_train)

		if 'verbose' in self.kwargs:
			train_pred = self.model.predict(self.x_train)
			print(f"Model type: {self.model.__repr__()}. An evaluation report from train data\n{cls_report(self.y_train, train_pred)}")
	def evaluate(self):
		test_pred = self.model.predict(x=self.x_test)

		print(f"Model type: {self.model.__repr__()}. An evaluation report from test data\n{cls_report(self.y_test, test_pred)}")