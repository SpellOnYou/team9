from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from mlp import MlpModel
from sklearn.metrics import classification_report

def get_path_text():
    path_d = Path('../datasets/emotions/isear/text-based')
    path_train = path_d / 'train_val_osp_tense.csv'
    path_test = path_d / 'test_osp_tense.csv'

    return path_train, path_test

def get_path_rule():
    path_d = Path('../datasets/emotions/isear/rule-driven')
    path_train = path_d / 'train_val_occ_rule.csv'
    path_test = path_d / 'test_occ_rule.csv'

    return path_train, path_test


def get_data(data_path):
    label_ls, text_ls = [], []
    with data_path.open() as f:
        for line_idx, line in enumerate(f):
            if line_idx == 0:
                pass
            else:
                one_line = line.strip('\n\'\"\"').split(',', maxsplit=2)
                label_ls.append(one_line[1])
                try:
                    input_data = one_line[2].rsplit(',', maxsplit=3)
                except IndexError:
                    continue
                text_ls.append(' \t '.join(input_data))

    return label_ls, text_ls


def convert_one_hot(array):
    '''get 1-d array and convert to 2-d one-hot encoding'''
    mat = np.zeros((array.shape[0], array.max() + 1))
    mat[range(array.shape[0]), array] = 1
    return mat


def train_occ(x_train_, y_train_, x_test_, y_test_):
    '''

    Initialises the model
    Converts y_train to one hot encoded vector
    Fits model
    Predicts
    Prints classification report

    '''

    mlp_model = MlpModel()
    model = mlp_model.get_model(x_train_.shape[1])  # init model
    y_train_2d = convert_one_hot(y_train_)  # convert y label to 2-d
    mlp_model.fit_model(model, x_train_, y_train_2d)  # train
    pred = model.predict(x=x_test_, batch_size=64, verbose=1).argmax(-1)  # predict and get maximum value's index
    print(classification_report(y_test_, pred, target_names=sorted(list(label_to_idx.keys()))))  # report results


def get_input_combinations(text_ls, rule_or_text):

    '''
    This function returns all possible input combinations as a list.
    Input combinations rule-based file:

        [text, tense]
        [text, direction]
        [text, polarity]

        [text, tense, direction]
        [text, tense, polarity]
        [text, direction, polarity]

        [text, tense, direction, polarity]

    Input combinations text-based file:

        [text, osp]
        [text, tense]
        [text, osp, tense]

    '''

    rule_comb_1 = []
    rule_comb_2 = []
    rule_comb_3 = []
    rule_comb_4 = []
    rule_comb_5 = []
    rule_comb_6 = []
    rule_comb_7 = []

    text_comb_1 = []
    text_comb_2 = []
    text_comb_3 = []
    for input_data in text_ls:
        tmp = input_data.split('\t')
        #if len(tmp) == 4:
        if rule_or_text == 'rule':
            text = tmp[0]
            tense = tmp[1]
            direction = tmp[2]
            polarity = tmp[3]
#                text_ls.append(' \t '.join(input_data).strip('\n\"\"\"'))
            rule_comb_1.append(text+'\t'+tense)
            rule_comb_2.append(text+'\t'+direction)
            rule_comb_3.append(text+'\t'+polarity)
            rule_comb_4.append(text+'\t'+tense+'\t'+direction)
            rule_comb_5.append(text+'\t'+tense+'\t'+polarity)
            rule_comb_6.append(text+'\t'+direction+'\t'+polarity)
            rule_comb_7.append(text+'\t'+tense+'\t'+direction+'\t'+polarity)

        #elif len(tmp) == 3:
        elif rule_or_text == 'text':
            text = tmp[0]
            osp = tmp[1]
            tense = tmp[2]

            text_comb_1.append(text+'\t'+osp)
            text_comb_2.append(text+'\t'+tense)
            text_comb_3.append(text+'\t'+osp+'\t'+tense)

    if rule_or_text == 'rule':
        return [rule_comb_1, rule_comb_2, rule_comb_3, rule_comb_4, rule_comb_5, rule_comb_6, rule_comb_7]
    elif rule_or_text == 'text':
        return [text_comb_1, text_comb_2, text_comb_3]


# Get paths for text- and rule-based occ variables
train_path_text, test_path_text = get_path_text()
train_path_rule, test_path_rule = get_path_rule()

# load plain data from text based file
y_train_raw_text, x_train_raw_text = get_data(train_path_text)
y_test_raw_text, x_test_raw_text = get_data(test_path_text)

# load plain data from rule based file
y_train_raw_rule, x_train_raw_rule = get_data(train_path_rule)
y_test_raw_rule, x_test_raw_rule = get_data(test_path_rule)



# Get combinations for rule-based occ
rule_combinations_train = get_input_combinations(x_train_raw_rule, 'rule')
rule_combinations_test = get_input_combinations(x_test_raw_rule, 'rule')

# Get combinations for text-based occ
text_combinations_train = get_input_combinations(x_train_raw_text, 'text')
text_combinations_test = get_input_combinations(x_test_raw_text, 'text')

# convert to tfidf, for text data
# since default transform type is csr.sparse, need to be converted np.array
# TODO: there might be config arg in fit_transform or TfidfVectorizer, find out

vectorizer = TfidfVectorizer()
x_train = vectorizer.fit_transform(rule_combinations_train[6]).toarray() # [6] =  [text, direction, polarity]
x_test = vectorizer.transform(rule_combinations_test[6]).toarray()

# This is for rule-based only
# I have to do text-based tomorrow
# make target/label look-up table && recast to np array of int
label_to_idx = {label: idx for idx, label in enumerate(sorted(set(y_train_raw_rule)))}
idx_to_label = {v: k for k, v in label_to_idx.items()}
y_train = np.array(list(map(lambda x: label_to_idx[x], y_train_raw_rule)))
y_test = np.array(list(map(lambda x: label_to_idx[x], y_test_raw_rule)))

train_occ(x_train, y_train, x_test, y_test)

