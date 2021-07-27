# utils.py

def validate_y(labels):
	"""return referential dict when y label is not integer"""
	label2idx = dict()
	if not all(map(lambda x: isinstance(x, int), labels)):
		label2idx = {v:i for i, v in enumerate(set(labels))}
		labels = map(lambda x: label2idx[x], labels)

	return label2idx, list(labels)