"""classify.py; this is core implementation of this library"""

from .model import (
	mlp,
	lstm_awd,
	nb,)

__all__ = ["Classifier"]

class Classifier():
	def __init__(self, model=nb, emb_type= tfidf, occ_type='', *args, **kwargs):
		self.model = model,
		self.emb_type = emb_type
		self.occ_type = ''

	def get_data(self):
		datasets.read_csv()
