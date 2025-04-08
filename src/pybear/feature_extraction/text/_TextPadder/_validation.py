# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Sequence

import numbers

from ..__shared._validation._2D_X import _val_2D_X
from ..__shared._validation._any_integer import _val_any_integer
from ..__shared._validation._any_string import _val_any_string



def _validation(
    _X: Sequence[Sequence[str]],
    _fill:str,
    _n_features: numbers.Integral
) -> None:

    """
    Centralized hub for validating parameters and X.
    See the individual validation modules for more details.


    Parameters
    ----------
    _X:
        Sequence[Sequence[str]] - the data.
    _fill:
        str - the fill value for the void space in the data.
    _n_features:
        numbers.Integral - the number of features for the filled data.


    Return
    ------
    -
        None


    """


    _val_2D_X(_X, _require_all_finite=False)

    _val_any_string(_fill, 'fill', _can_be_None=False)

    _val_any_integer(
        _n_features, 'n_features', _min=0, _can_be_bool=False, _can_be_None=False
    )








