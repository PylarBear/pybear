# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Sequence
from typing_extensions import TypeAlias, Union

import numpy as np

from .....base._check_1D_str_sequence import check_1D_str_sequence
from .....base._check_2D_str_array import check_2D_str_array

XContainer: TypeAlias = Union[Sequence[str], Sequence[Sequence[str]]]


# pizza this needs test!


def _delete_empty_rows(
    X: XContainer
) -> XContainer:

    """
    Remove textless rows from data.


    Parameters
    ----------
    X:
        Union[Sequence[str], Sequence[Sequence[str]]] - the data.


    Return
    ------
    -
        X: Union[Sequence[str], Sequence[Sequence[str]]] - the data with
        empty rows removed.


    """


    # pizza there needs to be a decision here. might need to impose that
    # only list[str], list[list[str]], NDArray[str] be passed, given
    # all the various slicing that would have to be wrote out.

    # PIZZA ALSO SOMETIME WHEN U HAVE TIME CONVERT THIS TO REGEX
    # .*[A-Za-z0-9].*

    _is_list_of_lists = False
    try:
        check_1D_str_sequence(X)
    except:
        try:
            check_2D_str_array(X)
            _is_list_of_lists = True
        except:
            raise TypeError(
                f"'X' must be a vector of strings or an array of strings, "
                f"got {type(X)}."
            )



    if _is_list_of_lists:
        for row_idx in range(len(X) - 1, -1, -1):
            for EMPTY_OBJECT in [[''], [' '], ['\n'], ['\t'], []]:
                if np.array_equal(X[row_idx], EMPTY_OBJECT):
                    X = np.delete(X, row_idx, axis=0)
                    break
    else:
        # MUST BE list_of_strs
        for row_idx in range(len(X) - 1, -1, -1):
            if X[row_idx] in ['', ' ', '\n', '\t']:
                X = np.delete(X, row_idx, axis=0)


    return X






