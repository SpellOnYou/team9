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
                        "--model_type",
                        help="A model used for classification, \nNOTE: case insensitive",
                        metavar='MLP, NB, LSTM_AWD',
                        type=str,
                        default='NB')

    parser.add_argument("-t",
                        "--type",
                        help="A type of embedding,\n",
                        metavar='tfidf, fasttext',
                        type=str,
                        default='tfidf')
    parser.add_argument("-o",
                        "--occ_type",
                        help="A type of occ features,\nNOTE: It will use text only if no argument is given.",
                        metavar='rule, data',
                        type=str,
                        default='')
    parser.add_argument("-e",
                        "--emb_size",
                        help="A size of embedding dimension,\nNOTE: This is only valid when you use predict-based embedding, which is fasttext in our library.",
                        metavar='50 100 300',
                        type=str,
                        default='')
    parser.add_argument("-v",
                        "--verbose",
                        help=".",
                        # type=str, this is flag
                        default=False)                            

    # TODO: validate given arguments
    args = parser.parse_args()
    if False: parser.error(f'')

    return args

def main():
    args = get_args()
    print(args)
    # import pudb; pudb.set_trace()
    # team9.model.dummy()
    emo_cls = team9.Classifier(**args.__dict__)
    emo_cls()
    #TODO: 1) load datasets instance
    #TODO: 2) get embedding 'emb_type'
    #TODO: 3) load model and train with embedding
    #TODO: 4) interpret/analyse