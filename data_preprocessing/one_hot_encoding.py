# Author: Lara Grimminger

# load libraries
import numpy as np
import csv


class OneHotEncoding:
    """
    A class used to return the output data as one hot encoded data

    Attributes
    ----------
    mapping_dict: dict
        Contains the unique target labels of the data set and their indices.

    file: file
        The file containing the data set.

    csv_file: dict
        Reads the file which is in csv format into a dictionary form.

    labels: list
        List of output data.

    target_labels: list
        Universal list of emotions.

    labels_dict: dict
        Dictionary of universal emotions and their one hot encodings.

    Methods
    -------
    get_unique_labels:
        Returns a list of unique target labels.

    mapping:
        Maps each target label to an index.
        Loops over each unique target label and creates arrays where all indices are 0 except for the index of
        the respective target label, which is 1, and appends arrays to a list.
        Calls method generate_dictionary(one_hot_encoded).
        Returns a list of one hot encoded emotions.

    generate_dictionary(one_hot_encoded):
        Receives one hot encoded list of target labels as parameter.
        Generates a dictionary where the target label is the key and the corresponding one hot encoded target label
        is the value.
        This dictionary serves as a universal dictionary for the train, validation and test data set to make sure
        that the respective target labels are one hot encoded in the same way.

    one_hot_encoding(encoded_dict=None):
        Returns the emotions as one hot encoded vectors.

    get_encoded_dict:
        Returns the universal dictionary.

    """
    def __init__(self, file_name):
        """
        file_name: str
            The name of the file
        """
        self.mapping_dict = {}
        self.file = open(file_name)
        self.csv_file = csv.DictReader(self.file)
        self.labels = []
        for col in self.csv_file:
            self.labels.append(col["label"]) # column label

        self.target_labels = []
        for word in self.labels:
            if word not in self.target_labels:
                self.target_labels.append(word) # Appends only the unique emotions

        self.labels_dict = {}
        self.mapping()

    def get_unique_labels(self):
        """
        Returns universal list of emotions.
        """
        return self.target_labels

    def mapping(self):
        """
        Returns a list of universal one hot encoded emotions.
        """
        one_hot_encoded = []
        for label_idx in range(len(self.target_labels)):
            self.mapping_dict[self.target_labels[label_idx]] = label_idx

        for e in self.target_labels:
            binary_vec = [0] * len(self.target_labels) # list of zeros
            binary_vec[self.mapping_dict[e]] = 1 # index of emotion is 1
            one_hot_encoded.append(binary_vec)
        self.generate_dictionary(one_hot_encoded)

        return one_hot_encoded

    def generate_dictionary(self, one_hot_encoded):
        """
        Generates a dictionary where the unique target labels are the keys and their corresponding one hot encodings
        are the values.

        Parameter
        ----------
        one_hot_encoded: list
            List of universal one hot encoded emotions.
        """
        self.labels_dict = dict(zip(self.target_labels, one_hot_encoded))  # the universal dictionary

    def one_hot_encoding(self, encoded_dict=None):
        """
        Represents the target labels of the file as one hot encoded and returns them in a vector.
        If the argument `encoded_dict` isn't passed in, the universal dictionary is generated.

        Parameter
        ----------
        encoded_dict: dict
            The universal dictionary of the target labels of the train data set.
            Default is None.
            If None, the internally generated dictionary is used.
        """
        df_labels = []
        if encoded_dict is None:
            encoded_dict = self.labels_dict
        for e in self.labels:
            if e in encoded_dict.keys():
                df_labels.append(encoded_dict[e])
        return np.array(df_labels)

    def get_encoded_dict(self):
        """
        Returns the universal dictionary of emotions containing the corresponding one hot encoding of the emotions.
        """
        return self.labels_dict

#
# train_file = "../datasets/emotions/isear/isear-train-modified.csv"
# val_file = "../datasets/emotions/isear/isear-val-modified.csv"
# test_file = "../datasets/emotions/isear/isear-test-modified.csv"
#
# ohe_train = OneHotEncoding(train_file)
# ohe_val = OneHotEncoding(val_file)
# ohe_test = OneHotEncoding(test_file)
#
# y_train = ohe_train.one_hot_encoding()
# print("Y_train:\n", y_train)
# print(y_train.shape)
# reference_dict = ohe_train.get_encoded_dict()
#
# y_val = ohe_val.one_hot_encoding(reference_dict)
# print("Y_val:\n", y_val)
# print(y_val.shape)
#
# y_test = ohe_test.one_hot_encoding(reference_dict)
# print("Y_test:\n", y_test)
# print(y_test.shape)

