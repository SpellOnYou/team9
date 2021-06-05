# -*- coding: utf-8 -*-
# Author: Jiwon Kim, Lara Grimminger

# load standard library
from pathlib import Path
from torch import tensor, randn, zeros
import sys

#import modules need for MLP
from models.loss import CrossEntropy
from models.linear import Linear
from models.relu import Relu
from models.fscore import Fscore

# import text preprocessing module
from data_preprocessing.one_hot_encoding import OneHotEncoding



class MLP:
    """Make fully-connected multilayer perceptron.


    """	
    def __init__(self, **kwargs):
    	self.kwargs = {k:v for k, v in kwargs.items()}

    	self.is_trace = self.kwargs['trace'] if 'trace' in self.kwargs else False

  		# assign n_layer when user input exists
    	self.n_layers = self.kwargs['n_layers'] if 'n_layers' in self.kwargs else 2

    def __call__(self):

    	self._get_path()

    	print("Loading dataset....")
    	self._get_data()


    	self._get_model()

    	self.run()

    def	run(self):
    	pass
	
	def._get_model(self):
		pass

    def _get_path(self):

    	if 'root_data' in self.kwargs:
    		self._get_user_path(self.kwargs['root_path'])

		else:
			self.root_data = Path('datasets/emotions/isear')
    		self.train_path = self.data_path / 'isear-train-modified.csv'
    		self.valid_path = self.data_path / 'isear-valid-modified.csv'
    		self.test_path = self.data_path / 'isear-test-modified.csv'

	def _get_data(self):
		pml_train = PadMaxLength(train_path)
		pml_val = PadMaxLength(val_path)
		pml_test = PadMaxLength(test_path)

		bow_train = BagOfWords(pml_train.text)  # Sentences to create the vocabulary

		tf_idf_train = bow_train.tf_idf(pml_train.text)
		tf_idf_val = bow_train.tf_idf(pml_val.text, train=False)
		tf_idf_test = bow_train.tf_idf(pml_test.text, train=False)		

   
    def _get_user_path(self, path):
    	user_path = Path(path)
    	assert user_path.is_dir(), "root of data should be folder"

    	# Now we suppose use input is confiend to isear dataset. Since our model isn't flexible for various input.
    	# Not implemented yet but could be used (e.g. model test)

if __name__ == "__main__":
	# get keyward parameters (if exists)
	mlp = MLP(**dict(arg.split('=') for arg in sys.argv[1:]))
	
	print(mlp.n_layers)

