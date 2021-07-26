# dataset.base.py

__all__ = ['get_bytestring', 'load_csv']


from functools import partial
import io
import pkgutil
import pandas as pd

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
