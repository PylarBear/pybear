# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score
)


master_scorer_dict = {
    'accuracy': accuracy_score,
    'balanced_accuracy': balanced_accuracy_score,
    'average_precision': average_precision_score,
    'f1': f1_score,
    'precision': precision_score,
    'recall': recall_score
}








