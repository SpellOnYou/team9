# -*- coding: utf-8 -*-
# Author: Jiwon Kim

# before executing this code, if you don't have flair library, I recommend you to download it first using pip install flair.

from collections import Counter
from nltk.tokenize import word_tokenize, sent_tokenize
from flair.models import SequenceTagger
from flair.data import Sentence
from collections import defaultdict
import pandas as pd
import pathlib,	pickle,	nltk

#load pos from flair library
tagger = SequenceTagger.load('pos')


nltk.download('punkt') #this is for tokenization

# define combinations of tense and its tense. 0: past, 1: future, 2:present, 3:unknown
tense_comb_dict = {
	1:{
		('VB',): 2,
		('VBD',): 0,
		('VBG',): 0,
		('VBN',): 2,
		('VBP',): 2,
		('VBZ',): 2,
	},
	2:{
		('VB', 'MD'): 3,
		('VB', 'VBG'): 2,
		('VB', 'VBN'): 0,
		('VB', 'VBP'): 2,
		('VBD', 'VB'): 0,
		('VBD', 'VBG'): 0,
		('VBD', 'VBN'): 0,
		('VBD', 'VBP'): 0,
		('VBG', 'VB'): 2,
		('VBG', 'VBN'): 3,
		('VBG', 'VBP'): 2,
		('VBP', 'VB'): 2,
		('VBP', 'VBG'): 2,
		('VBP', 'VBN'): 2,
		('VBZ', 'VB'): 3,
		('VBZ', 'VBD'): 0,
		('VBZ', 'VBG'): 2,
		('VBZ', 'VBN'): 2,
		('VBZ', 'VBP'): 2

	},

	3:{
		('MD', 'VB', 'VBD'): 0,
		('MD', 'VB', 'VBG'): 3,
		('MD', 'VB', 'VBN'): 0,
		('MD', 'VB', 'VBP'): 2,
		('MD', 'VB', 'VBZ'): 2,
		('MD', 'VBD', 'VBG'): 0,
		('MD', 'VBG', 'VBP'): 2,
		('VB', 'VBD', 'VBG'): 3,
		('VB', 'VBD', 'VBN'): 0,
		('VB', 'VBD', 'VBP'): 3,
		('VB', 'VBD', 'VBZ'): 3,
		('VB', 'VBG', 'VBN'): 0,
		('VB', 'VBG', 'VBP'): 2,
		('VB', 'VBG', 'VBZ'): 2,
		('VB', 'VBN', 'VBP'): 3,
		('VB', 'VBN', 'VBZ'): 3,
		('VB', 'VBP', 'VBZ'): 2,
		('VBD', 'VBG', 'VBN'): 3,
		('VBD', 'VBG', 'VBP'): 3,
		('VBD', 'VBG', 'VBZ'): 3,
		('VBD', 'VBN', 'VBP'): 3,
		('VBD', 'VBN', 'VBZ'): 3,
		('VBD', 'VBP', 'VBZ'): 3,
		('VBG', 'VBN', 'VBP'): 0,
		('VBG', 'VBN', 'VBZ'): 2,
		('VBG', 'VBP', 'VBZ'): 2,
		('VBN', 'VBP', 'VBZ'): 2
	},
}

def get_data(path):
	with path.open() as f:
		for line_i, line in enumerate(f):
			if line_i == 0: continue
			label, text = line.split(',', maxsplit=1)
			yield label, text.strip('\n')

def get_tokens(trg_path):
	tokens_in_sents = []
	for y, text in get_data(trg_path):
		tokens_in_sents += [word_tokenize(text)] #insert unpacked list
	return tokens_in_sents

def save_pos(trg_tokens, trg_file):

	'''
	Description
	---
	get part of speech of all given tokens and save it at a given path.
		if no target file is given, it is saved at current path

	Note
	---
	This function takes time (I think flair is rather slow), so I recommend you to have sufficient time if you want to run retrieve all part of speechs for data.


	'''

	with open(trg_file, 'wb') as f:
		pos_dict = defaultdict()
		for sent_id, sent in enumerate(trg_tokens):
			sent = Sentence(sent)
			tagger.predict(sent)
			pos_dict[sent_id] = [pos.tag for pos in sent.get_spans('pos')]
			if sent_id % 100 == 0: print(f"{(sent_id)/len(trg_tokens)} done")
		pickle.dump(pos_dict, file = f)

def get_verb(pos_path):
	'''Get part of speech file and return unique set of verb types
	'''
	with pos_path.open('rb') as r:
		pos_list = pickle.load(r)

	filter_verb = lambda x: 1 if x == 'MD' or x.startswith('V') else 0

	# verb_list = defaultdict(list)
	verb_set = defaultdict(list)

	for sent_id, text in pos_list.items():
		# filter verb and model in part of speech sets
		# train_verb_list[sent_id] = list(filter(filter_verb, text))
		verb_set[sent_id] = list(set(filter(filter_verb, text)))

	return verb_set

def save_tense(pos_path, trg_path=None, mode = 'train'):
	'''
	Parameter
	---
	pos_path {pathlib or str}: file path where part of speech exists.
	trg_path {pathlib or str}: file path where output has to be saved.
	'''

	data_dict= {
		'train': train_path,
		'valid': valid_path,
		'test': test_path
	}

	tokens = get_tokens(data_dict[mode])

	if not pos_path.exists(): save_pos(tokens, pos_path)

	tense_column = defaultdict(int)
	verb_set = get_verb(pos_path)

	# here key is sentence id and value is combination of tenses
	for k, v in verb_set.items():
		# see if the number of comb is in out budget.
		if len(v) in tense_comb_dict.keys():
			if tuple(v) in tense_comb_dict[len(v)]:
				tense_column[k] = tense_comb_dict[len(v)][tuple(v)]
			else: tense_column[k]=3
		#unknwon if not exist

		else: tense_column[k] = 3

	if not trg_path: trg_path = crt_dir/f'{mode}_occ_tense_encoding.csv'

	#make sure the order of sents aren't mixed
	tense_results = [tense_column[k] for k in range(len(tokens))]
	pd.DataFrame(tense_results).to_csv(trg_path)	



root_dir = pathlib.Path('/content/CLab21/')
data_path = root_dir/'emotions/isear'
train_path = data_path / 'isear-train-modified.csv'
valid_path = data_path / 'isear-val-modified.csv'
test_path = data_path / 'isear-test-modified.csv'

train_pos = root_dir/'advanced_model/occ/train-part-of-speech.pkl'
valid_pos = root_dir/'advanced_model/occ/valid-part-of-speech.pkl'
test_pos = root_dir/'advanced_model/occ/test-part-of-speech.pkl'

crt_dir = pathlib.Path().resolve()

#saeve tense
save_tense(train_pos, trg_path, mode ='train')
save_tense(valid_pos, trg_path, mode ='valid')
save_tense(test_pos, trg_path, mode ='test')