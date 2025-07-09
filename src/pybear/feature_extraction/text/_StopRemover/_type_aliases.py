# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Sequence
import numpy.typing as npt
from typing_extensions import (
    TypeAlias,
    Union
)

import pandas as pd
import polars as pl



PythonTypes: TypeAlias = Sequence[Sequence[str]]
NumpyTypes: TypeAlias = npt.NDArray
PandasTypes: TypeAlias = pd.DataFrame
PolarsTypes: TypeAlias = pl.DataFrame

XContainer: TypeAlias = Union[PythonTypes, NumpyTypes, PandasTypes, PolarsTypes]

XWipContainer: TypeAlias = list[list[str]]

RowSupportType: TypeAlias = npt.NDArray[bool]






