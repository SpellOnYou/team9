from pathlib import Path
from lstm_awd import LstmAwd


class MainLSTM:

    def __init__(self, **kwargs):

        self.kwargs = {k: v for k, v in kwargs.items()}
        self.is_trace = self.kwargs['trace'] if 'trace' in self.kwargs else False

        self.file_names = ["LSTM_AWD_Text_confusion_matrix.png",
                           "LSTM_AWD_OCC_confusion_matrix.png",
                           "LSTM_AWD_Text_OCC_confusion_matrix.png"]

    def __call__(self):

        if self.is_trace:
            print("Loading dataset....")
        self.get_path()
        self.lstm_model = LstmAwd()
        self.train_lstm()

    def get_path(self):

        """Get data path, as we wrote down in self._get_user_path, this isn't currently user-interactive.
        """

        if 'root_data' in self.kwargs:
            self._get_user_path(self.kwargs['root_path'])

        else:
            self.data_path = Path('../datasets/emotions/isear')
            self.train_path = self.data_path / 'train_occ.csv'
            self.valid_path = self.data_path / 'val_occ.csv'
            self.test_path = self.data_path / 'test_occ.csv'

    def train_lstm(self):

        train_df, test_df = self.lstm_model.get_df(self.train_path, self.valid_path, self.test_path)

        # LSTM_AWD with feature text
        #model = self.lstm_model.train(train_df, 1, self.file_names[0])
        #self.lstm_model.test(model, self.test_path, ["text"])

        # LSTM_AWD with feature OCC variables
        model = self.lstm_model.train(train_df, [2, 3], self.file_names[1])
        self.lstm_model.test(model, self.test_path, ["osp", "tense"])

        # LSTM_AWD with features text and OCC variables
        #model = self.lstm_model.train(train_df, [1, 2, 3], self.file_names[2])
        #self.lstm_model.test(model, self.test_path, ["text", "osp", "tense"])


if __name__ == "__main__":
    lstm_model = MainLSTM()
    lstm_model()
