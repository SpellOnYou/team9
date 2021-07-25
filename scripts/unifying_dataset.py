'''
Author: jiwon kim
unifying_dataset.py
this script (maybe) won't be reused.
'''

data_root = Path('/content/CLab21/teamg9/datasets/isear')

def load_data():
    all_data = {}
    for occ_dir in data_root.iterdir():
        if occ_dir.is_dir():
            for occ_file in sorted(list(occ_dir.iterdir())):
                if occ_file.is_file() and not occ_file.stem.startswith('.'):
                    print(occ_file.stem)
                    all_data[occ_file.stem] = pd.read_csv(occ_file)
    return all_data
all_data = load_data()

def unify_process(df):
    df['text'] = df.text.apply(lambda x: re.sub("\"\"", "\"", x))
    df['text'] = df.text.apply(lambda x: x.strip("\"").strip("\'").strip())
    df = df.drop_duplicates(subset=['text'])
    return df

all_data = load_data()
test_rule, test_text = all_data['test_occ_rule'], all_data['test_occ']

if 'Unnamed: 0' in test_rule.columns: test_rule.drop(columns = test_rule.columns[0], inplace = True) #remove double index column
# # column name unify
test_rule.columns = ['label', 'text', 'tense', 'direction', 'osp']

len(test_rule), len(test_text)

test_text2 = unify_process(test_text)
test_rule2 = unify_process(test_rule)

test_df = test_rule2.merge(test_text2, on=['label', 'text'], suffixes=['_rule', '_text'])
len(test_df)

# convert numbers to symbol
osp_str = {0:'xxpos', 1:'xxneg', 2:'xxneu'}
test_df['osp_text'] = test_df.osp_text.apply(lambda x: osp_str[x])


tense_str = {0: 'xxpas', 1:'xxpre', 2:'xxfut', 3: 'xxpre'}
test_df['tense_text'] = test_df.tense_text.apply(lambda x: tense_str[x])

test_df.head()