# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

def _validate_return_train_score(
        _return_train_score: bool
    ) -> bool:

    """

    Validate return_train_score, which indicates whether or not to
    score the train data during the grid search, only booleans allowed.
    The test data is always scored. Train scores for the different folds
    can be compared against  the test scores for anomalies.


    Parameters
    ----------
    _return_train_score:
        bool - whether or not to score the training data.


    Return
    ------
    -
        _return_train_score:
            bool: validated return_train_score


    """

    if not isinstance(_return_train_score, bool):
        raise TypeError(f"return_train_score must be True or False")

    return _return_train_score



