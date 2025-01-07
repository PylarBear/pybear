# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



# 25_01_06 this module will probably go away.
# nan_mask and inf_mask will be put into a module like 'require_all_finite()'


from ..utilities._nan_masking import nan_mask




def check_nan(
    X,
    allow_nan: bool = True
) -> None:

    """
    Look for any nan-like representations in X. Raise a ValueError if
    any are present, otherwise return None.


    Parameters
    ----------
    X:
        array-like of shape (n_samples, n_features) or (n_samples,) -
        the data to be searched for nan-like representations.
    allow_nan:
        bool - If nan-like values are not found, return None. If nan-like
        values are found and this parameter is set to True, return None;
        if this parameter is set to False raise a ValueError.


    Return
    ------
    -
        None


    """


    if not allow_nan and any(nan_mask(X)):
         raise ValueError(f"there are nan-like values in X")














