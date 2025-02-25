# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import numpy.typing as npt
from .._type_aliases import (
    XContainer,
    RegExpRemoveType,
    RegExpFlagsType,
    RowSupportType
)

import re
import numbers

import numpy as np



def _regexp_2D_core(
    _X: XContainer,
    _regexp_remove: RegExpRemoveType,
    _regexp_flags: RegExpFlagsType
) -> tuple[XContainer, RowSupportType]:

    """
    Remove unwanted strings from a 2D dataset using regular expressions.


    Parameters
    ----------
     _X:
        XContainer - the data.
    _regexp_remove:
        RegExpRemoveType - the regexp pattern(s) by which to identify
        unwanted strings for removal.
    _regexp_flags:
        RegExpFlagsType - flags for the regexp patterns.


    Return
    ------
    -
        tuple[list[list[str]], RowSupportType]: the data with unwanted
        strings removed and a boolean vector indicating which rows of
        the data were kept.


    """


    assert isinstance(_X, list)
    for _ in _X:
        assert isinstance(_, list)
        assert all(map(isinstance, _, (str for i in _)))
    assert isinstance(_regexp_remove, (str, re.Pattern, list))
    assert isinstance(_regexp_flags, (type(None), numbers.Integral, list))


    # _regexp_remove must be str, re.Pattern, list[Union[str, re.Pattern, False]]

    # convert re.fullmatch params to lists -- -- -- -- -- -- -- -- --
    if isinstance(_regexp_remove, (str, re.Pattern)):
        _remove = [_regexp_remove for _ in _X]
    elif isinstance(_regexp_remove, list):
        _remove = _regexp_remove
    else:
        raise Exception

    if isinstance(_regexp_flags, (type(None), numbers.Integral)):
        _flags = [_regexp_flags for _ in _X]
    elif isinstance(_regexp_flags, list):
        _flags = _regexp_flags
    else:
        raise Exception
    # END convert re.fullmatch params to lists -- -- -- -- -- -- -- --


    _row_support: npt.NDArray[bool] = np.ones(len(_X), dtype=bool)


    for _idx in range(len(_X)-1, -1, -1):

        if _remove[_idx] is False:
            continue

        # must convert flags to int() or
        # re TypeError: unsupported operand type(s) for &: 'NoneType' and 'RegexFlag'
        MASK = np.logical_not(list(map(
            re.fullmatch,
            (_remove[_idx] for _ in _X[_idx]),
            _X[_idx],
            ((_flags[_idx] or 0) for _ in _X[_idx])
        ))).astype(bool)

        _X[_idx] = np.array(_X[_idx])[MASK].tolist()


        if len(_X[_idx]) == 0:
            _row_support[_idx] = False
            _X.pop(_idx)


    del _remove, _flags


    return _X, _row_support
















