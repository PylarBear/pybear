# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from .._type_aliases import XContainer

from ....__shared._validation._any_string import _val_any_string
from ....__shared._validation._1D_X import _val_1D_X
from ....__shared._validation._2D_X import _val_2D_X



def _val_X_join_2D(
    _X: XContainer,
    _join_2D: str
) -> None:

    """
    Validate the text container. Must be 1D or 2D. 2D containers Can be
    ragged.


    Parameters
    ----------
    _X:
        XContainer - the data to be justified. 2D containers can be
        ragged.
    _join_2D:
        str - Ignored if the data is given as a 1D sequence. For 2D
        containers of strings, this is the character string sequence
        that is used to join the strings across rows. The single string
        value is used to join for all rows.

    """


    try:
        _val_2D_X(_X, _require_all_finite=True)
        # only check join_2D if _X is 2D, join_2D is ignored if data is 1D
        _val_any_string(_join_2D, 'join_2D', _can_be_None=False)
    except:
        try:
            _val_1D_X(_X, _require_all_finite=True)
        except:
            raise TypeError(
                f"TextJustifier expected a 1D sequence of strings or a "
                f"(possibly ragged) 2D array-like of strings. See the docs "
                f"for clarification of accepted containers."
            )







