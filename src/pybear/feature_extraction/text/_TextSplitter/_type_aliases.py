# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import (
    TypeAlias
)
import numpy.typing as npt

import re
import numbers

import pandas as pd
import polars as pl



PythonTypes: TypeAlias = list[str] | tuple[str] | set[str]

NumpyTypes: TypeAlias = npt.NDArray[str]

PandasTypes: TypeAlias = pd.Series

PolarsTypes: TypeAlias = pl.Series

XContainer: TypeAlias = PythonTypes | NumpyTypes | PandasTypes | PolarsTypes

XWipContainer: TypeAlias = list[list[str]]

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

SepType: TypeAlias = \
    None |str | re.Pattern[str] | tuple[str | re.Pattern[str], ...]

SepsType: TypeAlias = SepType | list[SepType]

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

CaseSensitiveType: TypeAlias = bool | list[None | bool]

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

MaxSplitType: TypeAlias = numbers.Integral | None
MaxSplitsType: TypeAlias = MaxSplitType | list[MaxSplitType]

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

FlagType: TypeAlias = numbers.Integral | None
FlagsType: TypeAlias = FlagType | list[FlagType]





