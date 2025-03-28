# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Callable, Literal, Optional, Sequence
from typing_extensions import TypeAlias, Union
import numpy.typing as npt

import numbers
import re

import pandas as pd
import polars as pl



PythonTypes: TypeAlias = Union[Sequence[str], Sequence[Sequence[str]]]

NumpyTypes: TypeAlias = npt.NDArray[str]

PandasTypes: TypeAlias = Union[pd.Series, pd.DataFrame]

PolarsTypes: TypeAlias = Union[pl.Series, pl.DataFrame]

XContainer: TypeAlias = Union[PythonTypes, NumpyTypes, PandasTypes, PolarsTypes]

XWipContainer: TypeAlias = Union[list[str], list[list[str]]]

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

# str.replace(self, old, new, count=-1, /)
ReplaceType: TypeAlias = Callable[[str, str, Optional[numbers.Integral]], str]

OldType: TypeAlias = str
NewType: TypeAlias = str
CountType: TypeAlias = numbers.Integral

StrReplaceArgsType: TypeAlias = Union[
    tuple[OldType, NewType],
    tuple[OldType, NewType, CountType]
]

TRStrReplaceArgsType: TypeAlias = \
    Union[StrReplaceArgsType, set[StrReplaceArgsType]]

StrReplaceType: TypeAlias = Union[
    TRStrReplaceArgsType,
    list[Union[TRStrReplaceArgsType, Literal[False]]],
    None
]

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

# re.sub(pattern, repl, string, count=0, flags=0)
PatternType: TypeAlias = Callable[[str, Optional[numbers.Integral]], re.Pattern]
SearchType: TypeAlias = Union[str, PatternType]
ReplType: TypeAlias = Union[str, Callable[[re.Match], str]]
CountType: TypeAlias = numbers.Integral
FlagsType: TypeAlias = numbers.Integral

ReSubType: TypeAlias = Callable[
    [SearchType, ReplType, str, Optional[CountType], Optional[FlagsType]],
    str
]

RegExpReplaceArgsType: TypeAlias = Union[
    tuple[SearchType, ReplType],
    tuple[SearchType, ReplType, CountType],
    tuple[SearchType, ReplType, CountType, FlagsType],
]

TRRegExpReplaceArgsType: TypeAlias = \
    Union[RegExpReplaceArgsType, set[RegExpReplaceArgsType]]

RegExpReplaceType: TypeAlias = Union[
    TRRegExpReplaceArgsType,
    list[Union[TRRegExpReplaceArgsType, Literal[False]]],
    None
]








