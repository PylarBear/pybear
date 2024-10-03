# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#




from .._type_aliases import DataType, ColumnsType

import pandas as pd



def _header_handling_pre(
    _X:DataType,
    _columns:ColumnsType
):

    if _columns is not None:
        return _columns
    elif isinstance(_X, pd.core.frame.DataFrame):
        return _X.columns
    else:
        return _columns



