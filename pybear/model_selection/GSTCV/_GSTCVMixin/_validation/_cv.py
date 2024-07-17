# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from typing import Union, Iterable

from model_selection.GSTCV._type_aliases import GenericKFoldType




def _validate_cv(_cv: Union[None, int, GenericKFoldType]) -> \
    Union[None, int, GenericKFoldType]:

    """
    
    cv - int, None, or an iterable, default=None

    Determines the cross-validation splitting strategy.

    For integer/None inputs, StratifiedKFold is used with sklearn, KFold
    is used with dask.

    These splitters are instantiated with shuffle=False so the splits
    will be the same across calls.

    For passed numbers, validation is performed to ensure the number is
    an integer greater than 1. For passed iterables, however, no validation
    is done beyond verifying that it is an iterable that contains
    iterables of iterables. GSTCV will catch out of range indices and
    raise an IndexError. Any validation beyond that is up to the user.


    Parameters
    ----------
    _cv: int, None, Iterable - Possible inputs for cv are:
        1) None, to use the default 5-fold cross validation,
        2) integer, to specify the number of folds in a (Stratified)KFold,
        3) An iterable yielding (train, test) splits as arrays of indices.


    Return
    ------
    -
        _cv: int, Iterable - validated cv input

    
    """


    _cv = 5 if _cv is None else _cv


    err_msg = (
        "Possible inputs for cv are: "
        "\n1) None, to use the default 5-fold cross validation, "
        "\n2) integer > 1, to specify the number of folds in a (Stratified)KFold, "
        "\n3) An iterable yielding a sequence of (train, test) split pairs "
        "as arrays of indices."
    )


    _is_iter = False
    _is_int = False
    try:
        iter(_cv)
        if isinstance(_cv, (dict, str)):
            raise Exception
        _is_iter = True
    except:
        try:
            float(_cv)
            if isinstance(_cv, bool):
                raise Exception
            if int(_cv) != _cv:
                raise Exception
            _cv = int(_cv)
            _is_int = True
        except:
            raise TypeError(err_msg)


    assert _is_iter is not _is_int



    if _is_iter:
        ctr = 0
        for thing in _cv:
            ctr += 1
            try:
                iter(thing)
                if not len(thing) == 2:
                    raise Exception
                for array in thing:
                    iter(array)
            except:
                raise TypeError(err_msg)

        if ctr == 0:
            raise ValueError(f"'cv' is an empty iterable")


    elif _is_int:
        if _cv < 2:
            raise ValueError(err_msg + f"\nGot {_cv} instead.")



    return _cv













