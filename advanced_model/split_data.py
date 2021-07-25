import pandas as pd
from sklearn.model_selection import train_test_split

train_val = "/home/lara/PycharmProjects/CLab21_local_workspace/datasets/emotions/isear/rule-driven/train_val_occ_rule.csv"
test = "/home/lara/PycharmProjects/CLab21_local_workspace/datasets/emotions/isear/rule-driven/test_occ_rule.csv"

train_df = pd.read_csv(train_val)
test_df = pd.read_csv(test)

data_occ_rule = train_df.append(test_df)
data_occ_rule = data_occ_rule[:3000]

train, test = train_test_split(data_occ_rule, test_size=0.2, shuffle=True)
print(train.shape)
print(type(train))

train = train.to_csv("/home/lara/PycharmProjects/CLab21_local_workspace/datasets/emotions/isear/rule-driven/train_val_occ_rule_v3.csv", index=False)
test = test.to_csv("/home/lara/PycharmProjects/CLab21_local_workspace/datasets/emotions/isear/rule-driven/test_occ_rule_v3.csv", index=False)
