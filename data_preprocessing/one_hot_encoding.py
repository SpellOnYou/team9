import numpy as np
import csv


class OneHotEncoding:
    def __init__(self, file_name):
        self.mapping_dict = {}
        #self.csv_file = pd.read_csv(file_name)
        #self.labels = self.csv_file["label"]

        self.file = open(file_name)
        self.csv_file = csv.DictReader(self.file)
        self.labels = []
        for col in self.csv_file:
            self.labels.append(col["label"])

        self.target_labels = []
        for word in self.labels:
            if word not in self.target_labels:
                self.target_labels.append(word)

        #self.target_labels = self.labels.unique()
        self.labels_dict = {}
        self.mapping()

    def get_unique_labels(self):

        return self.target_labels

    def mapping(self):
        ### map each emotion to an integer
        one_hot_encoded = []
        for label_idx in range(len(self.target_labels)):
            self.mapping_dict[self.target_labels[label_idx]] = label_idx
        #print(self.mapping_dict)

        for c in self.target_labels:
            arr = list(np.zeros(len(self.target_labels), dtype=int))
            arr[self.mapping_dict[c]] = 1
            one_hot_encoded.append(arr)

        self.generate_dictionary(one_hot_encoded)

        return one_hot_encoded

    def generate_dictionary(self, one_hot_encoded):
        self.labels_dict = dict(zip(self.target_labels, one_hot_encoded))  # universal dict

    def one_hot_encoding(self, encoded_dict=None):
        df_labels = []
        if encoded_dict is None:
            encoded_dict = self.labels_dict
        for c in self.labels:
            if c in encoded_dict.keys():
                df_labels.append(encoded_dict[c])
        return np.array(df_labels)

    def get_encoded_dict(self):
        return self.labels_dict


