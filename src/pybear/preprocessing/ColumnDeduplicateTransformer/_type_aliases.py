# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from typing import Iterable, Literal, TypeAlias
from typing_extensions import Union
import numpy.typing as npt
import pandas as pd
import scipy.sparse as ss


DataType: TypeAlias = Union[
    npt.NDArray,
    pd.core.frame.DataFrame,
    # pizza finalize with sparse
]

KeepType: TypeAlias = Union[
    Literal['first'],
    Literal['last'],
    Literal['random']
]

DoNotDropType: TypeAlias = Union[
    Iterable[str],
    Iterable[Union[int, float]],
    None
]

ColumnsType: TypeAlias = Union[
    Iterable[str],
    None
]









