"""classify.py; this is core implementation of this library"""
import importlib

from .model import (
	mlp,
	lstm_awd,
	nb,)

# from .emb import *

from .dataset import *

__all__ = ["Classifier"]

class Classifier():
	def __init__(self, model=nb, emb_type= 'tfidf', occ_type='', *args, **kwargs):
		self.model = model,
		self.emb_type = emb_type
		self.occ_type = ''

	def get_data(self):

		self.train_x, self.train_y = load_csv('example/train.csv')(occ_type = self.occ_type)
	def get_embedding(self):
		self.embed = importlib.import_module(f'.emb.{self.emb_type}', __package__)
		print(self.embed)
