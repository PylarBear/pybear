# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



def _val_n_features_in(_n_features_in: int) -> int:

    """
    Validate n_features_in_ is non-bool integer >= 1.


    Parameters
    ----------
    _n_features_in:
        int - the number of features in the data being fitted.


    Return
    ------
    -
        None


    """


    err_msg = f"n_features_in must be an integer >= 1"

    try:
        float(_n_features_in)
        if isinstance(_n_features_in, bool):
            raise Exception
        if not int(_n_features_in) == _n_features_in:
            raise Exception
    except:
        raise TypeError(err_msg)

    if not _n_features_in >= 1:
        raise ValueError(err_msg)

    del err_msg


    return _n_features_in





