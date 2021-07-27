"""classify.py; this is core implementation of this library"""
import importlib

from .model import (
	mlp,
	lstm_awd,
	nb)

from .dataset import *

from .emb import tfidf

from .utils import validate_y

from .interpret import cls_report

__all__ = ["Classifier"]


emb_dict = {'tfidf': tfidf()}

class Classifier():
	def __init__(self, model=nb, emb_type= 'tfidf', occ_type='', *args, **kwargs):
		self.model = model()
		self.emb_type = emb_type
		self.occ_type = ''
		self.embedding = emb_dict[self.emb_type]
		self.kwargs = {k: v for k, v in kwargs.items()}


	def get_data(self):
		train_data_fn = load_csv('example/train.csv')
		import pudb; pudb.set_trace()
		self.x_train_text, self.y_train_label = train_data_fn(occ_type = self.occ_type)

		self.x_valid_text, self.y_valid_label = load_csv('example/valid.csv')(occ_type = self.occ_type)
		self.x_test_text, self.y_test_label = load_csv('example/test.csv')(occ_type = self.occ_type)
		
		# change when labels aren't int
		self.label2idx, self.y_train = validate_y(self.y_train_label)
		self.y_valid = list(map(lambda x: self.label2idx[x], self.y_valid_label))
		self.y_test = list(map(lambda x: self.label2idx[x], self.y_test_label))

	def train(self, x, y):
		self.x_train = self.embedding.fit_transform(x)
		self.model.fit(self.x_train, y)
		train_pred = self.model.predict(self.x_train)
		return train_pred

	def __call__(self):
		self.get_data()
		
		#check performance on data which have trained
		if 'verbose' in self.kwargs:
			train_pred = self.train(self.x_train_text, self.y_train)
			print(f"Model type: {self.model.__repr__()}. An evaluation report from train data\n{cls_report(self.y_train, train_pred)}")

		x_test = self.embedding.transform(self.x_test_text)
		test_pred = self.model.predict(x_test)

		print(f"Model type: {self.model.__repr__()}. An evaluation report from test data\n{cls_report(self.y_test, test_pred)}")
