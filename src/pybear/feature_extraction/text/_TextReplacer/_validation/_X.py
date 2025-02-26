# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from .._type_aliases import XContainer

from ._1D_str_sequence import _val_1D_str_sequence
from ._2D_str_array import _val_2D_str_array


def _val_X(_X: XContainer) -> None:

    """
    Validate X. Must be 1D list-like of strings or (possibly ragged) 2D
    array-like of strings.


    Parameters
    ----------
    _X:
        XContainer - the data.


    Returns
    -------
    -
        None


    """

    try:
        _val_1D_str_sequence(_X)
    except:
        try:
            _val_2D_str_array(_X)
        except:
            raise TypeError(
                f"X must be a 1D list-like of strings or a 2D (possibly ragged) "
                f"array-like of strings."
            )


