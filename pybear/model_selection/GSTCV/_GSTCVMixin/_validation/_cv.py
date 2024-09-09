# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from typing import Union, Iterable

from ....GSTCV._type_aliases import GenericKFoldType



def _validate_cv(
        _cv: Union[None, int, Iterable[GenericKFoldType]]
    ) -> Union[int, Iterable[GenericKFoldType]]:

    """

    Validate that _cv is:
    1) None,
    2) an integer > 1, or
    3) an iterable of tuples, with each tuple holding a pair of iterables;
    the outer iterable cannot be empty, and must contain than one pair.


    Parameters
    ----------
    _cv:
        int, Iterable, None -
        Possible inputs for cv are:
        1) None, to use the default 5-fold cross validation,
        2) integer, must be 2 or greater, to specify the number of folds
        in a (Stratified)KFold,
        3) An iterable yielding (train, test) split indices as arrays.


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
        "as arrays of indices, with at least 2 pairs."
    )


    _is_iter = False
    _is_int = False
    try:
        iter(_cv)
        if isinstance(_cv, (dict, str)):
            raise Exception
        _is_iter = True
        _cv = list(_cv)
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

        elif ctr == 1:
            raise ValueError(err_msg)


    elif _is_int:
        if _cv < 2:
            raise ValueError(err_msg + f"\nGot {_cv} instead.")


    return _cv













