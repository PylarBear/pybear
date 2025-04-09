# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Optional, Sequence
from typing_extensions import TypeAlias, Union
import numpy.typing as npt

import numbers

import pandas as pd
import polars as pl



PythonTypes: TypeAlias = Union[Sequence[str], Sequence[Sequence[str]], set[str]]

NumpyTypes: TypeAlias = npt.NDArray

PandasTypes: TypeAlias = Union[pd.Series, pd.DataFrame]

PolarsTypes: TypeAlias = Union[pl.Series, pl.DataFrame]

XContainer: TypeAlias = Union[PythonTypes, NumpyTypes, PandasTypes, PolarsTypes]

XWipContainer: TypeAlias = Union[list[str], list[list[str]]]

NCharsType: TypeAlias = Optional[numbers.Integral]

BackfillSepType: TypeAlias = Optional[str]

Join2DType: TypeAlias = Optional[str]




