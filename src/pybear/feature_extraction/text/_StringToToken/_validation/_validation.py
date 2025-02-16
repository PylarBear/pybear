# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing_extensions import Union

import numbers

from ._maxsplit import _val_maxsplit
from ._pad import _val_pad
from ._sep import _val_sep



def _validation(
    _maxsplit: numbers.Integral,
    _pad: str,
    _sep: Union[str, None],
    _using_dask_ml_wrapper: bool
) -> None:

    """
    Centralized hub for validation. See the individual modules for
    details.


    Parameters
    ----------
    _maxsplit:
        numbers.Integral
    _pad:
        str
    _sep:
        str
    _using_dask_ml_wrapper:
        bool


    Return
    ------
    -
        None

    """


    _val_maxsplit(_maxsplit)

    _val_pad(_pad, _using_dask_ml_wrapper)

    _val_sep(_sep)









