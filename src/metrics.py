"""
the function get a (y_true, y_pred) set of pairs and print the metrics for each problem
can be segmentation or classificaton
"""
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score

def classification_metrics(y_true, y_pred):

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    print("METRICS RESULTS:\n")
    print(f"\tAccuracy: {accuracy}")
    print(f"\tF1-score: {f1}")
    print(f"\tRecall: {recall}")