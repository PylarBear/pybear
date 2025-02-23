# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import numbers



def _val_n_features(_n_features: numbers.Integral) -> None:

    """
    Validate n_features. Must be a positive integer.


    Parameters
    ----------
    _n_features:
        numbers.Integral - the number of features (columns) in the final
        filled array.


    Return
    ------
    -
        None

    """


    err_msg = f"'n_features' must be a positive integer"

    if not isinstance(_n_features, numbers.Integral):
        raise TypeError(err_msg)

    if isinstance(_n_features, bool):
        raise TypeError(err_msg)

    if _n_features < 0:
        raise ValueError(err_msg)






