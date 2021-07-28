"""classify.py; this is core implementation of this library"""
import importlib

from sklearn.naive_bayes import MultinomialNB as nb
from .model import *
from .data import *
from .interpret import *
from .classify import *
from .utils import validate_y
from pathlib import Path

__all__ = ["Classifier"]

class Classifier():
	def __init__(self, model=nb, emb_type= 'tfidf', occ_type='', *args, **kwargs):
		self.model = model()
		self.emb_type = emb_type
		self.occ_type = occ_type
		self.kwargs = {k: v for k, v in kwargs.items()}


	# def get_data(self):
	# 	"""
	# 	Description:
	# 		here pkgname is module where data exists.

	# 	"""
	# 	self.x_train_text, self.y_train_label = load_csv(fpath = 'example/train', occ_type = self.occ_type)
	# 	self.x_valid_text, self.y_valid_label = load_csv(fpath= 'example/valid', occ_type = self.occ_type)
	# 	self.x_test_text, self.y_test_label = load_csv(fpath='example/test', occ_type = self.occ_type)
 #        # TODO: getting occ variable
	# 	# change when labels aren't int
	# 	self.label2idx, self.y_train = validate_y(self.y_train_label)
	# 	self.y_valid = list(map(lambda x: self.label2idx[x], self.y_valid_label))
	# 	self.y_test = list(map(lambda x: self.label2idx[x], self.y_test_label))

	# def get_embedding(self):
	# 	#TODO: it's better to divide fit / transform of tfidf and override function with additional class,
	# 	# so that both type of embedding can be obtained same method
	# 	if self.emb_type=='tfidf':
	# 		self.embedding = emb_dict[self.emb_type]
	# 		self.x_train = self.embedding.fit_transform(self.x_train_text)
	# 	else:
	# 		self.embedding = load_npz(self.emb_type)
	# 		self.vtoi = load_pkl(self.emb_type)

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
			train_pred = self.train(self.x_train_text, self.y_train)
			print(f"Model type: {self.model.__repr__()}. An evaluation report from train data\n{cls_report(self.y_train, train_pred)}")

		x_test = self.embedding.transform(self.x_test_text)
		test_pred = self.model.predict(x_test)

		print(f"Model type: {self.model.__repr__()}. An evaluation report from test data\n{cls_report(self.y_test, test_pred)}")
