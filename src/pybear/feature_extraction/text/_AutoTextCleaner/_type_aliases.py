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

MatchType: TypeAlias = Union[str, re.Pattern]

ReplaceType: TypeAlias = Union[str, Callable[[str], str]]

LexiconLookupType: TypeAlias = \
    Union[None, Literal['auto_add', 'auto_delete', 'manual']]

ReturnDimType: TypeAlias = Union[None, Literal['1D', '2D']]


class GetStatisticsType(TypedDict):

    before: Required[Union[None, bool]]
    after: Required[Union[None, bool]]









