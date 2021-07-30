# main.py
"""test classical nlp models with given features"""
import sys
import argparse
import team9

def get_args():
	"""Get command-line arguments"""

	# TODO: set the arguments: (inputs, emb_type, model)
	parser = argparse.ArgumentParser(
		description="",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter
	)
	parser.add_argument("-m",
						"--model_type",
						help="A model used for classification, \nNOTE: case insensitive",
						type=str,
						choices = ['NB', 'nb', 'mlp', 'MLP'],
						default='NB')
	parser.add_argument("-e",
						"--emb_type",
						help="A type of embedding\nNOTE: pretrained(predict-based) word embedding is not available to Naive Bayes, since, as you know, it assigns probability reffering frequency wherein negative value makes no sense.",
						type=str,
						choices = ['tfidf', 'fasttext'],
						default='tfidf')
	parser.add_argument("-o",
						"--occ_type",
						help="A type of occ features,\nNOTE: It will use text only if no argument is given.",
						type=str,
						choices = ['', 'rule' 'text'],
						default='')
	parser.add_argument("-d",
						"--dim",
						help="A size of embedding dimension.",
						type=str,
						choices = [50, 100, 300],
						default='50')
	parser.add_argument("-v",
						"--verbose",
						help="flags, if true, it will shows progress",
						action="store_true")

	# TODO: validate given arguments
	args = parser.parse_args()
	if False: parser.error(f'')

	return args

def main():
	try:
		args = get_args()
		#todo: main args and other optional args.. discrete
		# emo_cls = team9.Classifier(**args.__dict__)
		emo_cls = team9.Classifier(
			model_type=args.model_type,
			emb_type = args.emb_type,
			occ_type = args.occ_type,
			dim = args.dim,
			verbose = args.verbose
		)
		emo_cls()
		emo_cls.train()
		pred = emo_cls.predict()
		emo_cls.evaluate(emo_cls.y_test, pred)
	except (BrokenPipeError, IOError): pass
	
	sys.stderr.close()
