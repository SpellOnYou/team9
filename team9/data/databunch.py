# dataset.py

from .base import (
	tfidf,
	load_csv,
	load_npz,
	load_pkl,
	validate_y
	)

import numpy as np

# emb_type = {'pretrained': ['fasttext'], 'count': ['tfidf']}
emb_type = ['fasttext','tfidf']

class DataBunch():
	"""load all data and convert it to appropriate embedding type. embedding type will be obtrained in main(classify) module
	Note: when you load embedding from pre-trained model, it will have tensor size (dataset, max_length, emb_dim) which can't feed on MLP/NB.
		It will be flattend (i.e. tensor size (dataset, max_length * emb_dim)) unless other option is given.
	"""	
	def get_embedding(self, **kwargs):
		#TODO: it's better to divide fit / transform of tfidf and override function with additional class,
		# so that both type of embedding can be obtained same method

		assert self.kwargs['emb_type'] in emb_type, f"You've given currently unavailable embedding option : {self.kwargs['emb_type']},\nWe support {emb_type} only."

		# when embeddig type is count based
		if self.kwargs['emb_type']=='tfidf':
			self.embedder = tfidf()
			self.x_train = self.embedder.fit_transform(self.x_train_text)

		#when it's pretrained version
		else:
			self.embedder = self._load_pretrained
			self.x_train = np.array([self.embedder[pad_sent, ] for pad_sent in self.pad_num_sents])

# train_data = np.array([pre_trained[num_sent, ] for num_sent in self.num_pad])			

	def get_data(self):
		"""
		A function which loads dataset with given (or default) file path

		Attribute
		x_train_text: (list(one data))
			list of text data, and each sentence(or document) is not tokenized
		y_test_label: (list(one data))
		"""
		#TODO:we can get dir path's data at once and split by ourselves

		self.x_train_text, self.y_train_label = load_csv(fpath = 'example/train', occ_type = self.kwargs['occ_type'])
		self.x_valid_text, self.y_valid_label = load_csv(fpath= 'example/valid', occ_type = self.kwargs['occ_type'])
		self.x_test_text, self.y_test_label = load_csv(fpath='example/test', occ_type = self.kwargs['occ_type'])
		
		# change when labels aren't int with saving label's index
		self.label2idx, self.y_train = validate_y(self.y_train_label)
		self.y_valid = list(map(lambda x: self.label2idx[x], self.y_valid_label))
		self.y_test = list(map(lambda x: self.label2idx[x], self.y_test_label))

	@property
	def _load_pretrained(self, token='word', **kwargs):
		"""This function will 
			1. Tokenize according to tokenization type
				1.a. when it's word it uses tfidf vectorizer's tokenizer
					: Here we use same tokenizer which we used in tfidf, or other counter-based embedding
				1.b. when it's chracter it will use null space as delimeter
				Note: Other types of tokenizing (i.e., hybrid, byte, etc) vectors are not implemented yet
			2. Map tokenized words to indices of embedding and padding (if max_seq is given, sentence will be processed on that lenght)
			3. Rretrieve pretrained embedding
		"""
		
		#first process
		tokenizer = tfidf().build_tokenizer()
		tokens_in_sents = list(map(tokenizer, self.x_train_text))

		#second process
		#mapping to index
		self.vtoi = load_pkl(self.kwargs['emb_type'])
		num_fn = lambda tokens: [self.vtoi[token.lower()] if token.lower() in self.vtoi else -1 for token in tokens ]
		text_to_nums = map(num_fn, tokens_in_sents)

		#padding, vacant words will be substituted as unknown
		if not 'max_seq' in kwargs: max_seq = max([len(i) for i in tokens_in_sents])
		pad_fn = lambda sent: sent + [-1] * (max_seq-len(sent))
		self.pad_num_sents = list(map(pad_fn, text_to_nums))
		
		#third process
		pre_trained = load_npz(self.kwargs['emb_type'])
		return pre_trained
