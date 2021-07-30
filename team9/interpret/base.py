# team9.interpret.base
from lime.lime_text import LimeTextExplainer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report as cls_report
from sklearn.metrics import confusion_matrix
from pathlib import Path
from functools import partial
import seaborn as sn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def cm(true_label, pred_label, clf):
    """
    The function not only calculate confusion matrix, but also visualize (i.e. sys out write) as heatmap

    Parameters
    ----------
    true_label : array-like of shape (n_samples,)
        Ground truth of target data
    pred_label : array-like of shape (n_samples,)
        Predictoin from target data
    labels: array-like of shape (n_classes), default=None
        labels which index confusion matrix

    Note
    ----------
    Please be sure that input should be 1-D

    """
    cm_score = confusion_matrix(true_label, pred_label)
    df_cm = pd.DataFrame(cm_score, index=clf.labels, columns=clf.labels)
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True)
    img_path = Path(__package__.split('.')[0]) / 'interpret/results/cm'
    img_path.mkdir(exist_ok=True, parents=True)
    fname = img_path / f'ISEAR_{clf.model_type}_{clf.occ_type}'
    plt.savefig(fname)
    print(f"Your confusion matrix is saved at : {str(img_path)} named {fname}")

def lime_wrapper(embedder, model, labels, texts):
    """
    Parameters
    ----------
    embedder: A function which converts given texts to vectors (including one-hot)
    model: model which predict probability of input texts

    Return
    ----------
        probability of input text
    """
    embs = embedder.transform(texts)
    return model.predict(embs.toarray())

def lime(clf, trg_idx=None, test = True):
    """
    A function calls wrapped function and render adequate predictor using existing classifier.
    Since Lime gets only one text example, this function will get target index and handles lime stuff.
    Parameters
    ----------
    clf : {tema9.Classifier}

    trg_idx : {int}, default = randomly chosen index
        If no given index of target text, we randomly extract index depends on length of text data
    test : {bool}, default = True
        Assume target index is for the text data
    """
    
    lime_predictor = partial(lime_wrapper, clf.embedder, clf.learner, clf.labels)
    idx = trg_idx if trg_idx else np.random.choice(clf.x_test.shape[0])
    exp = LimeTextExplainer(class_names=clf.labels).explain_instance(clf.x_test_text[idx],lime_predictor, num_features=clf.c, top_labels=clf.c)
    exp_path = Path(__package__.split('.')[0]) / 'interpret/results/lime'
    exp_path.mkdir(exist_ok=True, parents=True)
    exp.save_to_file(exp_path / f"ISEAR_{clf.model_type}_{idx}_.html")
    exp.as_pyplot_figure().savefig(exp_path / f"ISEAR_{clf.model_type}_{idx}_.png")
    print(f"Your lime explanation is saved at : {str(exp_path)}")    