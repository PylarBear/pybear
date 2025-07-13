# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import (
    Optional,
    Sequence
)
from typing_extensions import (
    TypeAlias,
    Union
)
import numpy.typing as npt

import re
import numbers

import pandas as pd
import polars as pl



PythonTypes: TypeAlias = Union[Sequence[str], Sequence[Sequence[str]], set[str]]

NumpyTypes: TypeAlias = npt.NDArray[str]

PandasTypes: TypeAlias = Union[pd.Series, pd.DataFrame]

PolarsTypes: TypeAlias = Union[pl.Series, pl.DataFrame]

XContainer: TypeAlias = Union[PythonTypes, NumpyTypes, PandasTypes, PolarsTypes]

XWipContainer: TypeAlias = Union[list[str], list[list[str]]]

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

PatternType: TypeAlias = \
    Union[None, str, re.Pattern[str], tuple[Union[str, re.Pattern[str]], ...]]
RemoveType: TypeAlias = \
    Optional[Union[PatternType, list[PatternType]]]

WipPatternType: TypeAlias = \
    Union[None, re.Pattern[str], tuple[re.Pattern[str], ...]]
WipRemoveType: TypeAlias = \
    Optional[Union[WipPatternType, list[WipPatternType]]]

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

CaseSensitiveType: TypeAlias = Optional[Union[bool, list[Union[bool, None]]]]

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

RemoveEmptyRowsType: TypeAlias = Optional[bool]

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

FlagType: TypeAlias = Union[None, numbers.Integral]
FlagsType: TypeAlias = Optional[Union[FlagType, list[FlagType]]]

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

RowSupportType: TypeAlias = npt.NDArray[bool]




