## Author: Jiwon Kim

from torch import zeros
from torch import tensor
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

# get data
#======================make change the data path here===================================

src_path = Path('/content/CLab21/data/emotions/isear')
test_path = src_path/'isear-test-modified.csv'
train_path = src_path/'isear-train-modified.csv'
val_path = src_path/'isear-val-modified.csv'

#======================make change the data path here===================================

#======================this command was done in terminal================================
#!cut -f2- -d ',' {str(train_path)} > 'isear-train-modified-text'
#!cut -f2- -d ',' {str(val_path)} > 'isear-val-modified-text'
#!cut -f1 -d ',' {str(train_path)} > 'isear-train-modified-label'
#!cut -f1 -d ',' {str(val_path)} > 'isear-val-modified-label'
#======================this command was done in terminal================================

with open('isear-val-modified-text') as f:
    lines= f.readlines()
# remove header
val_text = lines[1:]


with open('isear-train-modified-text') as f:
    lines= f.readlines()
# remove header
train_text = lines[1:]


# load library method
tfidf = TfidfVectorizer(stop_words='english')
tfidf.fit_transform(train_text)
tfidf.fit_transform(val_text)

# slice by 3000 features
x_train = tfidf.fit_transform(train_text)[:, :3000]
x_valid = tfidf.fit_transform(val_text)[:, :3000]

# transform data type
x_train, x_valid = map(tensor, (x_train.todense(), x_valid.todense()))

class OneHotEncode():
	'''
	change label index to one-hot vector
	'''
    def __init__(self):
        self.label2idx = {}
        self.idx2label = {}
    def __call__(self, f_name, is_train=True):
        with open(f_name) as f:
            labels = f.read().lower().split('\n')[1:-1]
        
        if is_train:
            self.label2idx = {label: idx for idx, label in enumerate(set(labels)) if label}
            self.idx2label = {v:k for k, v in self.label2idx.items()}
        # convert to numeric variable
        labels = [self.label2idx[label] for label in labels]
        # make one-hot vector
        one_hot_vector = zeros(len(labels), len(set(labels)))
        one_hot_vector[range(one_hot_vector.shape[0]), labels] = 1
        return one_hot_vector

y_encode = OneHotEncode()
y_train= y_encode('isear-train-modified-label')
y_valid= y_encode('isear-val-modified-label', is_train=False)

# Now you can use (x_train, y_train, x_valid, y_valid)
