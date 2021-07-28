# dataset.py
pretrained = ['fasttext']
from sklearn.feature_extraction.text import TfidfVectorizer as tfidf
import nltk
import nltk.tokenize

class DataBunch():

	def __init__(self, emb_type='tfidf'):
		self.x, self.y = x, y

	def get_embedding(self):
		#TODO: it's better to divide fit / transform of tfidf and override function with additional class,
		# so that both type of embedding can be obtained same method
		if self.emb_type=='tfidf':
			self.embedding = emb_dict[self.emb_type]
			self.x_train = self.embedding.fit_transform(self.x_train_text)
		else:
			self.embedding = load_npz(self.emb_type)
			self.vtoi = load_pkl(self.emb_type)

	def get_data(self):
		"""
		text(list(str))
		label(list(str))
		"""
		#TODO:we can get dir path's data at once and split by ourselves

		self.x_train_text, self.y_train_label = load_csv(fpath = 'example/train', occ_type = self.occ_type)
		self.x_valid_text, self.y_valid_label = load_csv(fpath= 'example/valid', occ_type = self.occ_type)
		self.x_test_text, self.y_test_label = load_csv(fpath='example/test', occ_type = self.occ_type)
		
		# change when labels aren't int with saving label's index
		self.label2idx, self.y_train = validate_y(self.y_train_label)
		self.y_valid = list(map(lambda x: self.label2idx[x], self.y_valid_label))
		self.y_test = list(map(lambda x: self.label2idx[x], self.y_test_label))		

	# def __getitem__(self, idx):
	# 	return self.x[idx], self.y[idx]

