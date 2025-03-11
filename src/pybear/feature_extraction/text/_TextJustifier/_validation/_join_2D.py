# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Sequence
from typing_extensions import Union

import numbers

from .....base._check_1D_str_sequence import check_1D_str_sequence



def _val_join_2D(
    _join_2D: Union[str, Sequence[str]],
    _n_rows: numbers.Integral
) -> None:

    """
    Validate 'join_2D'. Must be a string or a 1D sequence of strings.


    Parameters
    ----------
    _join_2D:
        Union[str, Sequence[str]] - Ignored if the data is given as
        a 1D sequence. For 2D containers of (perhaps token) strings, the
        character string sequence(s) that are used to join the strings
        across rows. If a single string, that value is used to join for
        all rows. If a sequence of strings, then the number of strings
        in the sequence must match the number of rows in the data, and
        each entry in the sequence is applied to the corresponding entry
        in the data.
    _n_rows:
        numbers.Integral - the number of rows of text in the given data.


    Return
    ------
    -
        None

    """


    err_msg = (f"'join_2D' must be a string or a 1D sequence of strings. "
               f"\nif passed as a sequence, then the number of entries "
               f"must equal the number of rows in the data.")

    if isinstance(_join_2D, str):
        return

    try:
        check_1D_str_sequence(_join_2D)
    except:
        raise TypeError(err_msg)

    if len(_join_2D) != _n_rows:
        raise ValueError(err_msg)





