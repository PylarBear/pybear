# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Sequence

from .....base._check_1D_str_sequence import check_1D_str_sequence



def _val_X(_X: Sequence[str]) -> None:

    """

    Parameters
    ----------
    _X:
        Sequence[str] - A 1D sequence of strings that is, or can be
        converted to, a numpy array.


    Return
    ------
    -
        None


    """


    check_1D_str_sequence(_X)


    try:
        map(str, _X)
    except:
        raise TypeError(f"'X' must contain data that can be converted to str")









