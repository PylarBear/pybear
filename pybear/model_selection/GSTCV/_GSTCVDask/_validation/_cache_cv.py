# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#




def _validate_cache_cv(_cache_cv: bool) -> bool:

    """
    cache_cv can only be boolean. Indicates if the train/test fold
    indices are stored as lists once first generated, or if the indices
    are generated from scratch at each point of use.

    Parameters
    ----------
    _cache_cv:
        bool - to be validated

    Return
    ------
    -
        _cache_cv - validated _cache_cv


    """


    if not isinstance(_cache_cv, bool):
        raise TypeError(f'kwarg cache_cv must be a bool')

    return _cache_cv









