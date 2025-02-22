# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#
import numbers
from typing import Sequence
from typing_extensions import Union
from .._type_aliases import XContainer
import numpy.typing as npt

import itertools

import numpy as np



def _transform(
    _X: XContainer,
    _sep: Union[str, Sequence[str]],
    _maxsplit: Union[numbers.Integral, Sequence[numbers.Integral]]
    _fill: str
) -> npt.NDArray[str]:

    """
    Pad ragged X vector will fill value to make a full array.


    Parameters
    ----------
    _X:
        XContainer -
    _sep:
        Union[str, Sequence[str], None] - the string sequence(s) to split on. Ignored if
        the data is passed as a 2D array of tokenized strings.
    _fill:
        str - the string value to fill void space with.



    """

    _val_sep(_sep)
    _val_fill(_fill)






    return _X







