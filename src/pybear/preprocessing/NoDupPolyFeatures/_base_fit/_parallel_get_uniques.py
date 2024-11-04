# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import numpy as np

from pybear.preprocessing.NoDupPolyFeatures._type_aliases import DataType
from typing import Literal
from typing_extensions import Union
from numbers import Real

from joblib import wrap_non_picklable_objects


@wrap_non_picklable_objects
def _parallel_get_uniques(_X_column: DataType) -> Union[Real, Literal[False]]:

    _unqs = np.unique(_X_column, return_counts=False)

    # if len of uniques is 1, return the value
    if len(_unqs) == 1:
        return _unqs[0]
    # otherwise return False
    else:
        return False










