# team9.interpret.base
from lime.lime_text import LimeTextExplainer
from sklearn.metrics import classification_report as cls_report
from sklearn.metrics import confusion_matrix

from pathlib import Path
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt


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
    text:
        Variations of text instance

    Returns predicted label for the text instance generated from the model
    -------

    """
    vectorizer = TfidfVectorizer()
    vectorizer.fit_transform(x_data_train)
    text_vector = vectorizer.transform(text).toarray()
    prob = model.predict(text_vector, batch_size=64, verbose=1)
    return prob


def lime(x_data_train, x_data_test):

    """
    Creates an explainer object
    Calls method lime_predictor to receive prediction for text instance
    Generates an explanation with 7 features for text instance of the test set.

    Parameters
    ----------
    x_data_train: array-like
        Train sentences

    x_data_test: array-like
        Test sentences

    Returns explanation for text instance
    -------

    """
    idx = 906
    target_names = ["Anger", "Disgust", "Fear", "Guilt", "Joy", "Sadness", "Shame"]
    
    row = x_data_test[idx] # row of the test data whose prediction will be explained with LIME
    print("Row: %s" % row)
    exp = explainer.explain_instance(row, lime_predictor, num_features=7, top_labels=7)
    exp.save_to_file(f"results_Lime/ISEAR_{idx}_occ_rule.html")

    return exp


def lime_experimental():
    explainer = LimeTextExplainer(class_names=target_names)