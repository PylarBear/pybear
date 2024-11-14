# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from .._type_aliases import DataType


import numpy as np





def _parallel_constant_finder(
    _column: DataType
) -> dict[int, any]:


    # determine if is num or str
    _is_flt = False
    _is_str = False
    try:
        np.float64(_column)
        _is_flt = True
    except:
        _is_str = True



    # pizza have to deal w nan up in here


    if _is_str:







    # pizza, this needs to return False if not equal or a value
    return WHAT!?!?!?!
















