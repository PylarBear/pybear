# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Sequence

import numbers

from ._fill import _val_fill

from ._n_features import _val_n_features

from ...__shared._validation._2D_X import _val_2D_X



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


    # pizza set this one to False 25_04_02
    _val_2D_X(_X, _require_all_finite=False)

    _val_fill(_fill)

    _val_n_features(_n_features)








