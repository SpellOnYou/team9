import pandas as pd
from sklearn import preprocessing

from keras.utils import to_categorical

class OCCVariables:

    def __init__(self, file_name):
        self.df = pd.read_csv(file_name)

    def preprocess_occ_data(self, feature_input):

        if len(feature_input) == 1:
            for value in feature_input:
                data = self.df[value]
                transformed_value = data.values # type numpy array
        else:
            df_mod = self.df[list(feature_input)]
            transformed_value = df_mod.values

        return transformed_value

    def preprocess_occ_rule(self, feature_input):
        le = preprocessing.LabelEncoder()
        target_names = []
        if len(feature_input) == 1:
            for value in feature_input:
                data = self.df[value]
                for d in data:
                    target_names.append(d)
                transformed_value = le.fit_transform(target_names)
                transformed_value = transformed_value.reshape(-1, 1)

        else:
            df_mod = self.df[list(feature_input)]
            df_col = list(df_mod.columns)
            for i in range(len(df_col)):
                df_mod[df_col[i]] = le.fit_transform(df_mod[df_col[i]])
            transformed_value = df_mod.values

        return transformed_value








