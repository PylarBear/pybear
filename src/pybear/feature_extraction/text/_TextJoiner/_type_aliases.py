# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Optional, Sequence
import numpy.typing as npt
from typing_extensions import (
    TypeAlias,
    Union
)

import pandas as pd
import polars as pl


PythonTypes: TypeAlias = Sequence[Sequence[str]]
NumpyTypes: TypeAlias = npt.NDArray[str]
PandasTypes: TypeAlias = pd.DataFrame
PolarsTypes: TypeAlias = pl.DataFrame

XContainer: TypeAlias = Union[PythonTypes, NumpyTypes, PandasTypes, PolarsTypes]

XWipContainer: TypeAlias = list[str]

SepType: TypeAlias = Optional[Union[str, Sequence[str]]]











