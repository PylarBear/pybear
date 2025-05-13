# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



def _val_cache_cv(
    _cache_cv: bool
) -> None:

    """
    'cache_cv' can only be boolean. Indicates if the train/test folds
    are to be stored once first generated, or if the folds are generated
    from X and y with the KFold indices at each point of use.


    Parameters
    ----------
    _cache_cv:
        bool - to be validated


    Return
    ------
    -
        _cache_cv: bool - validated boolean _cache_cv


    """


    if not isinstance(_cache_cv, bool):
        raise TypeError(f"'cache_cv' must be a bool")






