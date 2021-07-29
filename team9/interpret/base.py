# team9.interpret.base
from lime.lime_text import LimeTextExplainer
from sklearn.metrics import classification_report as cls_report
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from pathlib import Path
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from lime.lime_text import LimeTextExplainer

global X_TRAIN
global MODEL


def cm(true_label, pred_label, labels=None, fname='confusion_matrix', **kwargs):
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
    df_cm = pd.DataFrame(cm_score, index=labels, columns=labels)
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True, **kwargs)
    img_path = Path(__package__.split('.')[0]) / 'plot'
    img_path.mkdir(exist_ok=True, parents=True)
    plt.savefig(img_path / fname)
    print(f"Your confusion matrix is saved at : {str(img_path)} named {fname}")


def lime_predictor(text):

    """

    Parameters
    ----------
    text: {list}
        Variations of text input

    Returns predicted label for the text input generated from the model
    -------

    """
    global X_TRAIN # train data for fit and transform
    global MODEL # model for predicting
    vectorizer = TfidfVectorizer()
    vectorizer.fit_transform(X_TRAIN)
    text_vector = vectorizer.transform(text).toarray()
    prob = MODEL.predict(text_vector)
    return prob


def lime(x_data_train, x_data_test, model):

    """
    Creates an explainer object
    Calls method lime_predictor to receive prediction for text input
    Generates an explanation with 7 features for text input of the test data
    Saves explanations to .html file

    Parameters
    ----------
    model: model

    x_data_train: array-like
        Train sentences

    x_data_test: array-like
        Test sentences

    Returns explanation for text instance
    -------

    """
    global X_TRAIN  # train data for fit and transform
    global MODEL  # model for predicting
    idx = 906
    target_names = ["fear", "shame", "disgust", "anger", "guilt", "sadness", "joy"]
    explainer = LimeTextExplainer(class_names=target_names)
    row = x_data_test[idx] # row of the test data whose prediction will be explained with LIME
    print("Row: %s" % row)
    X_TRAIN = x_data_train
    MODEL = model
    exp = explainer.explain_instance(row, lime_predictor, num_features=7, top_labels=7)
    exp_path = Path(__package__.split('.')[0]) / 'interpret/lime_results'
    exp_path.mkdir(exist_ok=True, parents=True)
    exp.save_to_file(exp_path / f"ISEAR_{idx}_.html")
    print(f"Your lime explanation is saved at : {str(exp_path)}")

    return exp
