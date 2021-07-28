"""classify.py; this is core implementation of this library"""
import importlib

from sklearn.naive_bayes import MultinomialNB as nb
from .model import *
from .data import *
from .interpret import *
from .classify import *
from pathlib import Path

__all__ = ["Classifier"]

model_dict = {'nb': nb()}

class Classifier(DataBunch):
	def __init__(self, *args, **kwargs):
		super().__init__()
		self.kwargs = {k: v for k, v in kwargs.items()}
		self.model = model_dict[self.kwargs['model_type'].lower()]		

	def train(self, x, y):
		self.model.fit(x, y)
		train_pred = self.model.predict(x)
		return train_pred

	def __call__(self):
		# TODO Its better to split getting data and embedding to data module
		
		self.get_data()
		self.get_embedding()
		# import pudb; pudb.set_trace()
		self.train(self.x_train, self.y_train)

		#check performance on data which have trained
		if 'verbose' in self.kwargs:
			train_pred = self.train(self.x_train, self.y_train)
			print(f"Model type: {self.model.__repr__()}. An evaluation report from train data\n{cls_report(self.y_train, train_pred)}")

		x_test = self.embedder.transform(self.x_test_text)
		test_pred = self.model.predict(x_test)

		print(f"Model type: {self.model.__repr__()}. An evaluation report from test data\n{cls_report(self.y_test, test_pred)}")
