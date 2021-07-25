import pandas as pd
from pathlib import Path
import numpy as np

data_path = Path('../datasets/emotions/isear')
train_path = data_path / 'train_occ.csv'
valid_path = data_path / 'val_occ.csv'
test_path = data_path / 'test_occ.csv'

train_df, valid_df, test_df = map(pd.read_csv, (train_path, valid_path, test_path))

map_dict = {0: 'Positive', 1: 'Negative', 2: 'Neutral'}
train_df["osp_str"] = train_df["osp"].map(map_dict)
valid_df["osp_str"] = valid_df["osp"].map(map_dict)
test_df["osp_str"] = test_df["osp"].map(map_dict)

# col_titles = ["label", "osp_str", "text", "tense"]
# train_df = train_df.reindex(columns=col_titles)
# valid_df = valid_df.reindex(columns=col_titles)
# test_df = test_df.reindex(columns=col_titles)
#
train_df = train_df.rename(columns={'text': 'osp_str', 'osp_str': 'text'})
valid_df = valid_df.rename(columns={'text': 'osp_str', 'osp_str': 'text'})
test_df = test_df.rename(columns={'text': 'osp_str', 'osp_str': 'text'})

train_df.to_csv(data_path / 'train_occ_str.csv')
valid_df.to_csv(data_path / 'val_occ_str.csv')
test_df.to_csv(data_path / 'test_occ_str.csv')
#train_df, valid_df, test_df = map(pd.read_csv, (train_path, valid_path, test_path))
#for osp_var in train_df["osp"]:
        #osp_var.replace({0:"positive", 1:"negative", 2:"neutral"})




#
# train_df["text"] = train_df["text"].astype(str)
# train_df["osp"] = train_df["osp"].astype(str)
# train_df["tense"] = train_df["tense"].astype(str)
#
# valid_df["text"] = valid_df["text"].astype(str)
# valid_df["osp"] = valid_df["osp"].astype(str)
# valid_df["tense"] = valid_df["tense"].astype(str)
#
# test_df["text"] = test_df["text"].astype(str)
# test_df["osp"] = test_df["osp"].astype(str)
# test_df["tense"] = test_df["tense"].astype(str)
#
# train_df[train_df['osp'] == ''].index
# train_df[train_df['tense'] == ''].index
# train_df[train_df.osp.apply(lambda x:x.isspace() == False)] # will only return cases without empty spaces
# train_df[train_df.tense.apply(lambda x:x.isspace() == False)]
#
# valid_df[valid_df['osp'] == ''].index
# valid_df[valid_df['tense'] == ''].index
# valid_df[valid_df.osp.apply(lambda x:x.isspace() == False)] # will only return cases without empty spaces
# valid_df[valid_df.tense.apply(lambda x:x.isspace() == False)]
#
# test_df[test_df['osp'] == ''].index
# test_df[test_df['tense'] == ''].index
# test_df[test_df.osp.apply(lambda x:x.isspace() == False)] # will only return cases without empty spaces
# test_df[test_df.tense.apply(lambda x:x.isspace() == False)]
#
# print(train_df[~train_df.osp.str.contains('\w')].osp.count())
# print(train_df[~train_df.osp.str.contains('\w')].tense.count())
# print(valid_df[~valid_df.osp.str.contains('\w')].osp.count())
#
#
# print(valid_df[~valid_df.tense.str.contains('\w')].tense.count())
# print(test_df[~test_df.tense.str.contains('\w')].tense.count())
# print(test_df[~test_df.osp.str.contains('\w')].osp.count())
#
# DF_new_row=valid_df.loc[valid_df['tense']=='']
#
# df = valid_df.replace(' ', np.nan)  # to get rid of empty values
# nan_values = train_df[train_df.isna().any(axis=1)]  # to get all rows with Na


