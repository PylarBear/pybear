# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import (
    Callable,
    Literal,
    Sequence,
    TypedDict
)
from typing_extensions import (
    NotRequired,
    Required,
    TypeAlias,
    Union
)

import numpy.typing as npt

import re

import pandas as pd
import polars as pl



PythonTypes: TypeAlias = Union[Sequence[str], set[str], Sequence[Sequence[str]]]

NumpyTypes: TypeAlias = npt.NDArray[str]

PandasTypes: TypeAlias = Union[pd.Series, pd.DataFrame]

PolarsTypes: TypeAlias = Union[pl.Series, pl.DataFrame]

XContainer: TypeAlias = Union[PythonTypes, NumpyTypes, PandasTypes, PolarsTypes]

XWipContainer: TypeAlias = Union[list[str], list[list[str]]]

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

ReturnDimType: TypeAlias = Union[None, Literal[1, 2]]

FindType: TypeAlias = Union[str, re.Pattern[str]]
SubstituteType: TypeAlias = Union[str, Callable[[str], str]]
PairType: TypeAlias = tuple[FindType, SubstituteType]
ReplaceType: TypeAlias = Union[None, PairType, tuple[PairType, ...]]

RemoveType: TypeAlias = Union[None, FindType, tuple[FindType, ...]]

class LexiconLookupType(TypedDict):
    update_lexicon: NotRequired[bool]
    skip_numbers: NotRequired[bool]
    auto_split: NotRequired[bool]
    auto_add_to_lexicon: NotRequired[bool]
    auto_delete: NotRequired[bool]
    DELETE_ALWAYS: NotRequired[Union[Sequence[str], None]]
    REPLACE_ALWAYS: NotRequired[Union[dict[str, str], None]]
    SKIP_ALWAYS: NotRequired[Union[Sequence[str], None]]
    SPLIT_ALWAYS: NotRequired[Union[dict[str, Sequence[str]], None]]
    remove_empty_rows: NotRequired[bool]
    verbose: NotRequired[bool]

class NGramsType(TypedDict):
    ngrams: Required[Sequence[Sequence[FindType]]]
    wrap: Required[bool]

class GetStatisticsType(TypedDict):
    before: Required[Union[None, bool]]
    after: Required[Union[None, bool]]









