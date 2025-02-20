# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from ..._type_aliases import WIPXContainer

from ..._validation._1D_str_sequence import _val_1D_str_sequence
from ..._validation._2D_str_array import _val_2D_str_array



def _val_wip_X(
    _WIP_X: WIPXContainer
) -> None:


    try:
        _val_1D_str_sequence(_WIP_X)
    except:
        try:
            _val_2D_str_array(_WIP_X)
        except:
            raise TypeError(
                f"X must be a 1D vector of strings or a 2D array of "
                f"strings, got {type(_WIP_X)}."
            )








