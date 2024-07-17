# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

def _validate_return_train_score(_return_train_score: bool) -> bool:

    if not isinstance(_return_train_score, bool):
        raise TypeError(f"return_train_score must be True or False")

    return _return_train_score



