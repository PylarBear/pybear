# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import numpy as np


def _val_feature_names(
        _columns: np.ndarray,
        _feature_names_in: np.ndarray
    ) -> np.ndarray:

    """Validate feature names in the currently passed data
    against feature names extracted from data seen in previous fits.

    Parameters
    ----------
    _feature_names_in:
        np.ndarray[str] - Feature names extracted from X.
    
    Return
    ------
    -
        None 

    
    """

    try:
        iter(_columns)
        if isinstance(_columns, (str, dict)):
            raise Exception
    except:
        raise TypeError(f"_columns like be a list-like iterable of strings")

    try:
        iter(_feature_names_in)
        if isinstance(_feature_names_in, (str, dict)):
            raise Exception
    except:
        raise TypeError(
            f"_feature_names_in like be a list-like iterable of strings"
        )


    if not np.array_equiv(_columns, _feature_names_in):
        _base_msg = (f"Columns passed to partial_fit() or transform() must "
                     f"match the columns passed during previous fit(s). ")
        _not_seen = [__ for __ in _columns if __ not in _feature_names_in]
        _missing = [__ for __ in _feature_names_in if __ not in _columns]
        _msg_1 = ''
        if len(_not_seen) > 10:
            _msg_1 = (f"\n{len(_not_seen)} new columns that were not seen "
                      f"during the first fit.")
        elif len(_not_seen) > 0:
            _msg_1 = (f"\nNew columns not seen during first fit: "
                      f"\n{', '.join(map(str, _not_seen))}")
        _msg_2 = ''
        if len(_missing) > 10:
            _msg_2 = (f"\n{len(_missing)} original columns not seen during "
                      f"this fit.")
        elif len(_missing) > 0:
            _msg_2 = (f"\nOriginal columns not seen during this fit: "
                      f"\n{', '.join(map(str, _missing))}")

        raise ValueError(_base_msg + _msg_1 + _msg_2)










