# team9.interpret.base

from sklearn.metrics import classification_report as cls_report
from sklearn.metrics import confusion_matrix

from pathlib import Path
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

def cm(true_label, pred_label, labels=None, fname = 'confusion_matrix', **kwargs):
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
    df_cm = pd.DataFrame(cm_score, index = labels, columns = labels)
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True, **kwargs)
    img_path = Path(__package__.split('.')[0])/'plot'
    img_path.mkdir(exist_ok=True, parents = True)
    plt.savefig(img_path/fname)
    print(f"Your confusion matrix is saved at : {str(img_path)} named {fname}")
    
def lime(): pass