import pandas as pd

train_file = "/home/lara/PycharmProjects/pythonProject/CLab21/data/emotions/isear/isear-train.csv"
val_file = "/home/lara/PycharmProjects/pythonProject/CLab21/data/emotions/isear/isear-val.csv"
test_file = "/home/lara/PycharmProjects/pythonProject/CLab21/data/emotions/isear/isear-test.csv"

df_train = pd.read_csv(train_file, names=["label", "text"], header = None)
df_val = pd.read_csv(val_file, names=["label", "text"], header = None)
df_test = pd.read_csv(test_file, names=["label", "text"], header = None)


df_train_mod = df_train.dropna(axis=0, how='any')
df_val_mod = df_val.dropna(axis=0, how='any')
df_test_mod = df_test.dropna(axis=0, how='any')


print("Len original df", len(df_train))
print("Len modified df", len(df_train_mod))

print("Len original df", len(df_val))
print("Len modified df", len(df_val_mod))

print("Len original df", len(df_test))
print("Len modified df", len(df_test_mod))

df_train_mod.to_csv("/home/lara/PycharmProjects/pythonProject/CLab21/data/emotions/isear/isear-train-modified.csv", index=False)
df_val_mod.to_csv("/home/lara/PycharmProjects/pythonProject/CLab21/data/emotions/isear/isear-val-modified.csv", index=False)
df_test_mod.to_csv("/home/lara/PycharmProjects/pythonProject/CLab21/data/emotions/isear/isear-test-modified.csv", index=False)