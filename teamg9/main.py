# main.py
"""test classical nlp models with given features"""

import argparse

# ----------------
def get_args():
    """Get command-line arguments"""

    # TODO: set the arguments: (inputs, emb_type, model)
    parser = argparse.ArgumentParser()
    parser.add_argument()

    # TODO: validate given arguments
    args = parser.parse_args()
    if False: parser.error(f'')

    return args

def main():
    args = get_args()
    #TODO: 1) load datasets instance
    #TODO: 2) get embedding 'emb_type'
    #TODO: 3) load model and train with embedding
    #TODO: 4) interpret/analyse
