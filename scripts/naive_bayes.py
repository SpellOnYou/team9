# -*- coding: utf-8 -*-
"""naive_bayes.py

This file will be moved to models/ and splitted to 1) dataset load 2) text embedding 3) model
"""

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
# Equivalent to CountVectorizer followed by TfidfTransformer.
import operator
from sklearn.metrics import classification_report as f1_metric

get_combined_df = lambda x: [' '.join(i) for i in zip(x["text"], x["tense"].map(str), x["over-polarity"].map(str), x["direction"].map(str))]

def load_text(is_occ = True):
	# TODO: argument is_occ will be converted to 'feature_list' which substitutes `_feature_dict`
	'''load text from data_path, with/without occ module.'''
	feature_list = ['text', 'tense', 'over-polarity', 'direction'] if is_occ else: ['text']

	get_combined_df = lambda x: [pd.read_csv(f_i) for f_i in feature_list]

	return [' '.join(i) for i in zip(*get_combined_df)]


get_df = lambda x: pd.read_csv(x, index_col=0)
train_df, valid_df, test_df = map(get_df, (train_path, valid_path, test_path))
train_df = train_df.append(valid_df)


vectorizer = TfidfVectorizer()


"""### Text Only"""

x_train = vectorizer.fit_transform(train_df['text'])
x_test = vectorizer.transform(test_df['text'])
model = MultinomialNB()
model.fit(x_train, train_df['label'])

x_train
pred = model.predict(x_test)

print(f1_metric(pred, test_df['label']))

"""### Full OCC"""

train_occ = get_combined_df(train_df)
train_occ[:10]

test_occ = get_combined_df(test_df)

x_train = vectorizer.fit_transform(train_occ)
x_test = vectorizer.transform(test_occ)
model = MultinomialNB()
model.fit(x_train, train_df['label'])

x_train
pred = model.predict(x_test)

print(f1_metric(pred, test_df['label']))