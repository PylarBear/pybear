# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import (
    Callable,
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

FindType: TypeAlias = Union[str, re.Pattern[str]]
SubstituteType: TypeAlias = Union[str, Callable[[str], str]]
PairType: TypeAlias = tuple[FindType, SubstituteType]
ReplaceSubType: TypeAlias = Union[None, PairType, tuple[PairType, ...]]
ReplaceType: TypeAlias = Optional[Union[ReplaceSubType, list[ReplaceSubType]]]

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

WipPairType: TypeAlias = tuple[re.Pattern[str], SubstituteType]
WipReplaceSubType: TypeAlias = Union[None, WipPairType, tuple[WipPairType, ...]]
WipReplaceType: TypeAlias = \
    Optional[Union[WipReplaceSubType, list[WipReplaceSubType]]]

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

CaseSensitiveType: TypeAlias = Optional[Union[bool, list[Union[bool, None]]]]

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

FlagType: TypeAlias = Union[None, numbers.Integral]
FlagsType: TypeAlias = Optional[Union[FlagType, list[FlagType]]]







