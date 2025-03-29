# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Literal, Sequence
from typing_extensions import TypeAlias, Union
import numpy.typing as npt

import re
import numbers

import pandas as pd
import polars as pl



PythonTypes: TypeAlias = Union[Sequence[str], Sequence[Sequence[str]]]

NumpyTypes: TypeAlias = npt.NDArray[str]

PandasTypes: TypeAlias = Union[pd.Series, pd.DataFrame]

PolarsTypes: TypeAlias = Union[pl.Series, pl.DataFrame]

XContainer: TypeAlias = Union[PythonTypes, NumpyTypes, PandasTypes, PolarsTypes]

XWipContainer: TypeAlias = Union[list[str], list[list[str]]]

StrType: TypeAlias = Union[str, set[str]]
StrRemoveType: TypeAlias = \
    Union[None, StrType, list[Union[StrType, Literal[False]]]]

RegExpType: TypeAlias = Union[str, re.Pattern]
RegExpRemoveType: TypeAlias = \
    Union[None, RegExpType, list[Union[RegExpType, Literal[False]]]]

FlagType: TypeAlias = Union[None, numbers.Integral]
RegExpFlagsType: TypeAlias = \
    Union[FlagType, list[Union[FlagType, Literal[False]]]]

RemoveEmptyRowsType: TypeAlias = bool

RowSupportType: TypeAlias = npt.NDArray[bool]




