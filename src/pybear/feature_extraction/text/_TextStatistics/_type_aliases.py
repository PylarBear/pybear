# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import TypedDict
from typing_extensions import Required, TypeAlias, Union
import numpy.typing as npt

import numbers

import pandas as pd
import polars as pl



class OverallStatisticsType(TypedDict):

    size: Required[numbers.Integral]
    uniques_count: Required[numbers.Integral]
    average_length: Required[numbers.Real]
    std_length: Required[numbers.Real]
    max_length: Required[numbers.Integral]
    min_length: Required[numbers.Integral]



PythonTypes: TypeAlias = Union[list[str], tuple[str], set[str]]

NumpyTypes: TypeAlias = npt.NDArray[str]

PandasTypes: TypeAlias = pd.Series

PolarsTypes: TypeAlias = pl.Series

XContainer: TypeAlias = Union[PythonTypes, NumpyTypes, PandasTypes, PolarsTypes]

# pizza verify this
XWipContainer: TypeAlias = Union[list[str], list[list[str]]]





