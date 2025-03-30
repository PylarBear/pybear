# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Literal, Sequence
from typing_extensions import Union
import numpy.typing as npt
from .._type_aliases import (
    RegExpType,
    RegExpFlagsType,
    RowSupportType
)

import numpy as np

import re

from ._param_conditioner import _param_conditioner



def _regexp_1D_core(
    _X: Sequence[str],
    _regexp_remove: Union[RegExpType, list[Union[RegExpType, Literal[False]]]],
    _regexp_flags: RegExpFlagsType
) -> tuple[Sequence[str], RowSupportType]:

    """
    Remove unwanted strings from a 1D dataset using regular expressions.


    Parameters
    ----------
     _X:
        XContainer - the data.
    _regexp_remove:
        Union[RegExpType, list[Union[RegExpType, Literal[False]]]] - the
        pattern(s) by which to identify strings to be removed.
        _regexp_remove cannot be None. The code that allows entry
        into this module explicitly says "if regexp_remove is not None:".
    _regexp_flags:
        RegExpFlagsType - flags for the regexp patterns.


    Return
    ------
    -
        tuple[list[str], RowSupportType]: the data with unwanted strings
        removed and a boolean vector indicating which rows of data were
        kept.

    """


    # validation -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    assert isinstance(_X, list)
    assert all(map(isinstance, _X, (str for _ in _X)))
    # END validation -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --


    _remove, _flags = _param_conditioner(
        _regexp_remove,
        _regexp_flags,
        _n_rows=len(_X)
    )


    _row_support: npt.NDArray[bool] = np.ones(len(_X), dtype=bool)

    for _idx in range(len(_X)-1, -1, -1):

        if _remove[_idx] is False:
            continue

        # _regexp_remove aka _remove[_idx] must be Union[str, re.Pattern]
        if re.fullmatch(
            _remove[_idx],
            _X[_idx],
            **{'flags': _flags[_idx]} if _flags[_idx] is not None else {}
        ):
            _row_support[_idx] = False
            _X.pop(_idx)


    del _remove, _flags


    return _X, _row_support







