# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from .._type_aliases import DataType
from typing import Literal
from typing_extensions import Union

from numbers import Real

import numpy as np

from ....utilities import nan_mask_numerical, nan_mask_string




def _parallel_constant_finder(
    _column: DataType,
    _equal_nan: bool,
    _rtol: Real,
    _atol: Real
) -> Union[Literal[False], any]:


    # determine if is num or str
    _is_flt = False
    _is_str = False
    try:
        np.float64(_column)
        _is_flt = True
    except:
        _is_str = True



    # pizza have to deal w nan up in here

    if _is_flt:
        nan_mask_numerical()

    elif _is_str:

        nan_mask_string()





    # pizza, this needs to return False if not equal or a value
    return
















