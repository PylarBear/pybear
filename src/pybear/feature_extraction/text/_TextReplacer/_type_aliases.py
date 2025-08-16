# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import (
    Callable,
    Sequence,
    TypeAlias
)
import numpy.typing as npt

import re
import numbers

import pandas as pd
import polars as pl



PythonTypes: TypeAlias = Sequence[str] | Sequence[Sequence[str]] | set[str]

NumpyTypes: TypeAlias = npt.NDArray[str]

PandasTypes: TypeAlias = pd.Series | pd.DataFrame

PolarsTypes: TypeAlias = pl.Series | pl.DataFrame

XContainer: TypeAlias = PythonTypes | NumpyTypes | PandasTypes | PolarsTypes

XWipContainer: TypeAlias = list[str] | list[list[str]]

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

FindType: TypeAlias = str | re.Pattern[str]
SubstituteType: TypeAlias = str | Callable[[str], str]
PairType: TypeAlias = tuple[FindType, SubstituteType]
ReplaceSubType: TypeAlias = None | PairType | tuple[PairType, ...]
ReplaceType: TypeAlias = ReplaceSubType | list[ReplaceSubType]

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

WipPairType: TypeAlias = tuple[re.Pattern[str], SubstituteType]
WipReplaceSubType: TypeAlias = None | WipPairType | tuple[WipPairType, ...]
WipReplaceType: TypeAlias = WipReplaceSubType | list[WipReplaceSubType]

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

CaseSensitiveType: TypeAlias = bool | list[bool | None]

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

FlagType: TypeAlias = numbers.Integral | None
FlagsType: TypeAlias = FlagType | list[FlagType]







