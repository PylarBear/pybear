# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



def _val_n_features_in(_n_features_in: int) -> int:

    """
    Validate n_features_in_ is non-bool integer >= 1.

    """

    err_msg = f"n_features_in must be an integer >= 1"

    if isinstance(_n_features_in, bool):
        raise TypeError(err_msg)

    try:
        float(_n_features_in)
        if not int(_n_features_in) == _n_features_in:
            raise Exception
    except:
        raise TypeError(err_msg)

    if not _n_features_in >= 1:
        raise ValueError(err_msg)
    del err_msg

    return _n_features_in





