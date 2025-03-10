# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from .._type_aliases import XContainer

from .....base._check_2D_str_array import check_2D_str_array
from ._sep import _val_sep



def _validation(
    _X: XContainer,
    _sep: str
) -> None:

    """
    Centralized hub for validation. See the individual modules for
    details.


    Parameters
    ----------
    _X:
        XContainer - the (possibly ragged, perhaps tokenized) 2D text
        data to be joined along rows into a 1D list of strings.
    _sep:
        str - the string character to insert between strings in each row
        of the given 2D text data.


    Return
    ------
    -
        None

    """


    check_2D_str_array(_X, require_all_finite=True)

    _val_sep(_sep, _X)









