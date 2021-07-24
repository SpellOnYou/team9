from itertools import combinations

import pandas as pd
import csv
from sklearn import preprocessing
from pathlib import Path

isear_train_osp = "/home/lara/PycharmProjects/CLab21_local_workspace/datasets/emotions/isear/OCC_osp/train_occ_osp.csv"
isear_train_tense = "/home/lara/PycharmProjects/CLab21_local_workspace/datasets/emotions/isear/OCC_tense/train_occ_tense.csv"

isear_valid_osp = "/home/lara/PycharmProjects/CLab21_local_workspace/datasets/emotions/isear/OCC_osp/val_occ_osp.csv"
isear_valid_tense = "/home/lara/PycharmProjects/CLab21_local_workspace/datasets/emotions/isear/OCC_tense/val_occ_tense.csv"

isear_test_osp = "/home/lara/PycharmProjects/CLab21_local_workspace/datasets/emotions/isear/OCC_osp/test_occ_osp.csv"
isear_test_tense = "/home/lara/PycharmProjects/CLab21_local_workspace/datasets/emotions/isear/OCC_tense/test_occ_tense.csv"

output_train = "/home/lara/PycharmProjects/CLab21_local_workspace/datasets/emotions/isear/text-based/train_osp_tense.csv"
output_val = "/home/lara/PycharmProjects/CLab21_local_workspace/datasets/emotions/isear/text-based/val_osp_tense.csv"
output_test = "/home/lara/PycharmProjects/CLab21_local_workspace/datasets/emotions/isear/text-based/test_osp_tense.csv"
#
# #df_train = pd.read_csv(output_train)
#
# #df_meta = df[["tense", "osp"]]
# #X = df_meta.values
# # y = df['reviews_score']
#
# #print(X)
# # print(y)
#
#
a = pd.read_csv(isear_train_osp)
b = pd.read_csv(isear_train_tense)

merged_train = a.merge(b)

c = pd.read_csv(isear_valid_osp)
d = pd.read_csv(isear_valid_tense)

merged_val = c.merge(d)

e = pd.read_csv(isear_test_osp)
f = pd.read_csv(isear_test_tense)

merged_test = e.merge(f)

merged_train.to_csv(output_train, index=False)
merged_val.to_csv(output_val, index=False)
merged_test.to_csv(output_test, index=False)


# a = pd.read_csv("/home/lara/PycharmProjects/CLab21_local_workspace/datasets/emotions/isear/train_occ.csv")
# b = pd.read_csv("/home/lara/PycharmProjects/CLab21_local_workspace/datasets/emotions/isear/val_occ.csv")
test_df = pd.read_csv("/home/lara/PycharmProjects/CLab21_local_workspace/datasets/emotions/isear/test_occ.csv")
#
# print(type(test_df))
test_text_occ= test_df[["osp", "tense"]]
# #print(test_text_occ.head)
# test_df["osp"]= test_df["osp"].apply(str)
# print(test_df.head)

#total_rows['ColumnID'] = total_rows['ColumnID'].astype(str)

test_text_occ = test_text_occ.values
print(type(test_text_occ))
print("values", test_text_occ)

occ_rule = ['tense', 'direction', 'over-polarity']
features = list(list(combinations(occ_rule, i + 1)) for i, _ in enumerate(occ_rule))
print(features)


z = "/home/lara/PycharmProjects/CLab21_local_workspace/datasets/emotions/isear/rule-driven/test_occ_rule.csv"
test_text_occ_rule = pd.read_csv(z)
#test_text_occ_rule = test_text_occ_rule[["tense", "polarity"]]
le = preprocessing.LabelEncoder()

# df = test_text_occ_rule[["tense", "direction"]]
# df_col = list(df.columns)
# for i in range(len(df_col)):
#     df[df_col[i]] = le.fit_transform(df[df_col[i]])
# print(type(df.values))
# print(df.values)



# target_names = []  # list
# for dir, tense in test_text_occ_rule:
#     target_names.append(dir, tense)  # Appends labels to list
# le.fit(target_names)
# transformed_dir = le.transform(target_names)
# transformed_dir.to
feature_input = ("tense", "direction")



target_names_tense = []  # list

df_mod = test_text_occ_rule[list(feature_input)]

df_col = list(df_mod.columns)
for i in range(len(df_col)):
    df_mod[df_col[i]] = le.fit_transform(df_mod[df_col[i]])
transformed_value = df_mod.values

print("values", transformed_value)

# for variable in test_text_occ_rule.tense:
#     target_names_tense.append(variable)  # Appends labels to list
# le.fit(target_names_tense)
# transformed_tense = le.transform(target_names_tense)
# print(transformed_tense)

# test_rule = test_text_occ_rule.values
# print("shape", test_rule.shape)
# print(type(test_rule))
# print(test_rule)

#print(transformed_dir.reshape(-1, 1).shape)

#
# g = "/home/lara/PycharmProjects/CLab21_local_workspace/datasets/emotions/isear/rule-driven/train_occ_rule.csv"
# h = "/home/lara/PycharmProjects/CLab21_local_workspace/datasets/emotions/isear/rule-driven/valid_occ_rule.csv"
#
# gg = pd.read_csv(g)
# hh = pd.read_csv(h)

# #data.to_csv("/home/lara/PycharmProjects/CLab21_local_workspace/datasets/emotions/isear/test_occ.csv", index=False)
#
# train_df = gg.append(hh)
# train_df.to_csv("/home/lara/PycharmProjects/CLab21_local_workspace/datasets/emotions/isear/rule-driven/train_val_occ_rule.csv", index=False)

# a= '../datasets/emotions/isear/isear-train-modified-v2.csv'
# b = "/home/lara/PycharmProjects/CLab21/datasets/emotions/isear/isear-train-modified.csv"
# f = open(b)
#
#
# for line_i, line in enumerate(f):
#     if line_i == 0:
#         continue
#     label, text = line.split(',', maxsplit=1)
    #yield label.lower(), text.strip('\n')
        #, osp.strip('\n'), tense.strip('\n')


#
# if 'root_data' in kwargs:
#     self._get_user_path(self.kwargs['root_path'])
#
# else:
#     self.data_path = Path('../datasets/emotions/isear')
#     self.train_path = self.data_path / 'train-modified-v2.csv'
#     self.valid_path = self.data_path / 'val-modified-v2.csv'
#      self.test_path = self.data_path / 'test-modified-v2.csv'
