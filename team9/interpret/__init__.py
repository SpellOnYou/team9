# interpret.__init__.py

from sklearn.metrics import classification_report as cls_report

# @Lara / we can add lime here
__all__ = ["cls_report", "confusion_matrix", "lime"]

def confusion_matrix(): pass
def lime(): pass