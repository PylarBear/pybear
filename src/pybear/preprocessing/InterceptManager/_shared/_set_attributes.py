# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Iterable
from typing_extensions import Union
import numpy.typing as npt
import numpy as np



def _set_attributes(
    constant_columns_: dict[int, any],
    _instructions: dict[str, Union[None, Iterable[int], dict[str, any]]],
    _n_features: int
) -> tuple[dict[int, any], dict[int, any], npt.NDArray[bool]]:

    """
    use the constant_columns_ and _instructions attributes to build the
    kept_columns_, removed_columns_, and column_mask_ attributes.




    Parameter
    ---------
    constant_columns_: dict[int, any],
    _instructions: dict[str, Union[None, Iterable[int]]],
    _n_features: int


    Return
    ------
    -
        kept_columns_: dict[int: any], removed_columns_: dict[int, any], _column_mask_: NDArray[bool]

    """




    # validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
    assert isinstance(constant_columns_, dict)
    assert all(map(isinstance, constant_columns_, (int for _ in constant_columns_)))
    assert all(np.fromiter(constant_columns_, dtype=int) >= 0)
    assert all(np.fromiter(constant_columns_, dtype=int) <= _n_features - 1)
    assert isinstance(_instructions, dict)
    assert all([_ in ('keep', 'delete', 'add') for _ in _instructions])
    assert isinstance(_instructions['keep'], (type(None), list, np.ndarray))
    assert isinstance(_instructions['delete'], (type(None), list, np.ndarray))
    assert not any([c_idx in (_instructions['delete'] or []) for c_idx in (_instructions['keep'] or [])]), \
        f"column index in both 'keep' and 'delete'"
    assert isinstance(_instructions['add'], (type(None), dict))
    assert isinstance(_n_features, int)
    assert _n_features >= 0
    # END validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    kept_columns_: dict[int, any] = {}
    removed_columns_: dict[int, any] = {}
    # set column_mask_ dtype to object to carry py bool, not np.bool_.
    # unfortunately, numpy will not allow py bools in an object array for
    # slicing! so live with the yucky np.True_ and np.False_ repr.
    column_mask_ = np.ones(_n_features).astype(bool)

    # all values in _instructions dict are None could only happen if there are
    # no constant columns, in which case this for loop wont be entered
    for col_idx, constant_value in constant_columns_.items():
        if col_idx in (_instructions['keep'] or {}):
            kept_columns_[col_idx] = constant_value
        elif col_idx in (_instructions['delete'] or {}):
            removed_columns_[col_idx] = constant_value
            column_mask_[col_idx] = False
        else:
            raise Exception(
                f"a constant column in constant_columns_ is unaccounted for "
                f"in _instructions."
            )


    return kept_columns_, removed_columns_, column_mask_



