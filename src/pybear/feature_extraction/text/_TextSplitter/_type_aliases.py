# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Optional
from typing_extensions import (
    TypeAlias,
    Union
)
import numpy.typing as npt

import re
import numbers

import pandas as pd
import polars as pl



PythonTypes: TypeAlias = Union[list[str], tuple[str], set[str]]

NumpyTypes: TypeAlias = npt.NDArray[str]

PandasTypes: TypeAlias = pd.Series

PolarsTypes: TypeAlias = pl.Series

XContainer: TypeAlias = Union[PythonTypes, NumpyTypes, PandasTypes, PolarsTypes]

XWipContainer: TypeAlias = list[list[str]]

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

SepType: TypeAlias = Union[
    None,
    Union[str, re.Pattern[str]],
    tuple[Union[str, re.Pattern[str]], ...]
]
SepsType: TypeAlias = Optional[Union[SepType, list[SepType]]]

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

CaseSensitiveType: TypeAlias = Optional[Union[bool, list[Union[None, bool]]]]

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

MaxSplitType: TypeAlias = Union[None, numbers.Integral]
MaxSplitsType: TypeAlias = Optional[Union[MaxSplitType, list[MaxSplitType]]]

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

FlagType: TypeAlias = Union[None, numbers.Integral]
FlagsType: TypeAlias = Optional[Union[FlagType, list[FlagType]]]






















