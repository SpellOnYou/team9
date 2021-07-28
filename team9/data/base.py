# data.base.py
import pkgutil
from functools import partial
import io
import importlib
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer


def load_csv(fpath, occ_type='', index_col=0, **kwargs):
	"""
	since our data is string, we first approach the source data with pkgutil we need to make works in zip
	after reading in bytestring, we assume encoding type of source data is utf-8 (You may fix this with kwargs)
	and get data using pandas data frame object 

	Note
	---
		Some features don't have its value (as you can see in our report), we need to take care of None type

	return
	---
		x:
		y:
	"""
	pkgname = __package__
	if 'package' in kwargs: pkgname = kwargs['package']

	bytestring = pkgutil.get_data(pkgname, fpath)

	data = pd.read_csv(io.StringIO(bytestring.decode()), sep=',', index_col=0)
	data = data.fillna(value='')
	# import pudb; pudb.set_trace()

	cols = ['text']

	# add features if 
	
	if occ_type:
		cols.extend([f'osp_{occ_type}', f'tense_{occ_type}'])

	# get features which is in cols list
	features = map(lambda x: data.__getitem__(x).tolist(), cols)

	# unzip features and join with tab. one data is tuple(when variable is more than one)
	x = [' '.join(list(one_data))  for one_data in list(zip(*features))]
	y = data['label'].tolist()

	return x, y

# this is already zip_compresssed
# (and included to MANIFEST)format so that we don;t need to read it as bytestring
def load_npz(emb_type = 'fasttext', emb_dim=100):
	"""A function loads embedding"""
	# find current path
	embedding_path = Path(__file__).parent / f'pretrained/{emb_type}.en.{emb_dim}.npz'
	assert embedding_path.exists(), f"there is no version for {emb_type} with dimension size: {emb_dim}"
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
	#TODO: make this path rubust using pkgutil
	fname = Path(__file__).parent/f'pretrained/{emb_type}.en.vtoi.pkl'
	with fname.open('rb') as f:
		vtoi = pickle.load(f)
	return vtoi
