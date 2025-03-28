# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Literal
from typing_extensions import TypeAlias, Union

import re
import numbers

import pandas as pd
import polars as pl



PythonTypes: TypeAlias = Union[list[str], tuple[str], set[str]]

PandasTypes: TypeAlias = pd.Series

PolarsTypes: TypeAlias = pl.Series

XContainer: TypeAlias = Union[PythonTypes, PandasTypes, PolarsTypes]

XWipContainer: TypeAlias = list[list[str]]

SepType: TypeAlias = Union[str, set[str], None]
StrSepType: TypeAlias = Union[SepType, list[Union[SepType, Literal[False]]]]

RegExpType: TypeAlias = Union[str, re.Pattern]
RegExpSepType: TypeAlias = \
    Union[RegExpType, None, list[Union[RegExpType, Literal[False]]]]

MaxSplitType: TypeAlias = Union[numbers.Integral, None]
StrMaxSplitType: TypeAlias = \
    Union[MaxSplitType, list[Union[MaxSplitType, Literal[False]]]]
RegExpMaxSplitType: TypeAlias = \
    Union[MaxSplitType, list[Union[MaxSplitType, Literal[False]]]]

FlagType: TypeAlias = Union[numbers.Integral, None]
RegExpFlagsType: TypeAlias = \
    Union[FlagType, list[Union[FlagType, Literal[False]]]]






















