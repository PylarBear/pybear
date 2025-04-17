# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import (
    Callable,
    Literal,
    Optional,
    Sequence,
    TypedDict
)
from typing_extensions import Required, TypeAlias, Union
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
    update_lexicon: Optional[bool]
    skip_numbers: Optional[bool]
    skip_numbers: Optional[bool]
    auto_split: Optional[bool]
    auto_add_to_lexicon: Optional[bool]
    auto_delete: Optional[bool]
    DELETE_ALWAYS: Optional[Union[Sequence[str], None]]
    REPLACE_ALWAYS: Optional[Union[dict[str, str], None]]
    SKIP_ALWAYS: Optional[Union[Sequence[str], None]]
    SPLIT_ALWAYS: Optional[Union[dict[str, Sequence[str]], None]]
    remove_empty_rows: Optional[bool]
    verbose: Optional[bool]

class NGramsType(TypedDict):
    ngrams: Required[Sequence[Sequence[FindType]]]
    wrap: Required[bool]

class GetStatisticsType(TypedDict):
    before: Required[Union[None, bool]]
    after: Required[Union[None, bool]]









