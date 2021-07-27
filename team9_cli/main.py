# main.py
"""test classical nlp models with given features"""

import argparse
import team9

# ----------------
def get_args():
    """Get command-line arguments"""

    # TODO: set the arguments: (inputs, emb_type, model)
    parser = argparse.ArgumentParser(
        description="",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-m",
                        "--model",
                        help="A model used for classification, \nNOTE: case insensitive",
                        metavar='MLP, NB, LSTM_AWD',
                        type=str,
                        default='MLP')

    parser.add_argument("-e",
                        "--embedding",
                        help="An embedding type,\nNOTE: model LSTM_AWD will use its own pre-trained embedding.",
                        metavar='tfidf, fasttext',
                        type=str,
                        default='tfidf')
    parser.add_argument("-o",
                        "--occ",
                        help="A type of occ features,\nNOTE: It will use text only if no argument is given.",
                        metavar='rule, data',
                        type=str,
                        default='')
    
    # parser.add_argument("-l",
    #                     "--list",
    #                     help="A list of features given to model, \nNOTE: case insensitive",
    #                     metavar='MLP, NB, LSTM_AWD',
    #                     type=str,
    #                     default='MLP')

    # TODO: validate given arguments
    args = parser.parse_args()
    if False: parser.error(f'')

    return args

def main():
    args = get_args()
    print(args)
    # team9.model.dummy()
    emo_cls = team9.Classifier()
    emo_cls()
    #TODO: 1) load datasets instance
    #TODO: 2) get embedding 'emb_type'
    #TODO: 3) load model and train with embedding
    #TODO: 4) interpret/analyse