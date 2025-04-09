# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Optional, Sequence
from typing_extensions import TypeAlias, Union
import numpy.typing as npt

import numbers
import re

import pandas as pd
import polars as pl



PythonTypes: TypeAlias = Union[Sequence[str], Sequence[Sequence[str]], set[str]]

NumpyTypes: TypeAlias = npt.NDArray

PandasTypes: TypeAlias = Union[pd.Series, pd.DataFrame]

PolarsTypes: TypeAlias = Union[pl.Series, pl.DataFrame]

XContainer: TypeAlias = Union[PythonTypes, NumpyTypes, PandasTypes, PolarsTypes]

XWipContainer: TypeAlias = Union[list[str], list[list[str]]]

NCharsType: TypeAlias = Optional[numbers.Integral]

StrSepType: TypeAlias = Optional[Union[str, Sequence[str]]]

StrLineBreakType: TypeAlias = Optional[Union[str, Sequence[str], None]]

CaseSensitiveType: TypeAlias = Optional[bool]

RegExpSepType: TypeAlias = Optional[Union[re.Pattern[str], Sequence[re.Pattern[str]]]]

SepFlagsType: TypeAlias = Optional[Union[numbers.Integral, None]]

RegExpLineBreakType: TypeAlias = \
    Optional[Union[re.Pattern[str], Sequence[re.Pattern[str]], None]]

LineBreakFlagsType: TypeAlias = Optional[Union[numbers.Integral, None]]

BackfillSepType: TypeAlias = Optional[str]

Join2DType: TypeAlias = Optional[str]








