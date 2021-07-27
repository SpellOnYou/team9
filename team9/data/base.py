# data.base.py


from pathlib import Path
from functools import partial
import io
import pkgutil
import pandas as pd
import numpy as np
import pickle

def get_bytestring(formatter):
	"""function which reads data as a bytestring insensitive to cwd and packaging
	we can reuse this function when data file has different format/features/.."""
	def _inner(fpath='example/train.csv'):
		return partial(formatter, pkgutil.get_data( __package__ , fpath))
	return _inner

@get_bytestring
def load_csv(raw_data, occ_type='', index_col=0, **kwargs):
	
	"""get data using pandas data frame object
	here we assume encoding type of source data is utf-8 (You may fix this with kwargs)

	return
	---
		x:
		y:
	"""
	data = pd.read_csv(io.StringIO(raw_data.decode()), sep=',', index_col=0)

	cols = ['text']
	# add features if 

	if occ_type:
		cols.extend([f'osp_{occ_type}', f'tense_{occ_type}'])

	# get features which is in cols list
	features = map(lambda x: data.__getitem__(x).tolist(), cols)
	# unzip features and join with tab 
	x = ['\t'.join(one_data)  for one_data in list(zip(*features))]
	y = data['label'].tolist()

	return x, y

# this is already zip_compresssed
# (and included to MANIFEST)format so that we don;t need to read it as bytestring
def load_npz(emb_type = 'fasttext', emb_dim=100):
	"""A function loads embedding"""
	# find current path
	embedding_path = Path('.')/'pretrained/{emb_type}.en.{emb_dim}.npz'
	assert embedding_path.is_file(), f"there is no version for {emb_type} with dimension size: {dim_size}"
	with np.load(embedding_path) as ed:
		lookup_emb = ed[f'emb{emb_dim}']
	return lookup_emb

# this is also written in byte string
def load_pkl(emb_type='fasttext', **kwargs):
	"""A function for reading vocabulary to idx mapping dictionary
	return
	---
		collections.OrderedDict instance, key: word in dataset, value: idx
	"""
	with open(f'pretrained/{'.'}.en.vtoi.pkl' ,'rb') as f:
		vtoi = pickle.load(f)
	return vtoi