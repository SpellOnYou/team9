# dataset.py

class Dataset():
	def __init__(self):
		self.x, self.y = x, y
	def __getitem__(self, idx):
		return self.x[idx], self.y[idx]