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

import numbers
import re

import pandas as pd
import polars as pl



PythonTypes: TypeAlias = Sequence[Sequence[str]]

NumpyTypes: TypeAlias = npt.NDArray[str]

PandasTypes: TypeAlias = pd.DataFrame

PolarsTypes: TypeAlias = pl.DataFrame

XContainer: TypeAlias = Union[PythonTypes, NumpyTypes, PandasTypes, PolarsTypes]

XWipContainer: TypeAlias = list[list[str]]

NGramsType: TypeAlias = \
    Optional[Union[Sequence[Sequence[Union[str, re.Pattern[str]]]], None]]

NGramsWipType: TypeAlias = Union[None, list[tuple[re.Pattern[str], ...]]]

NGCallableType: TypeAlias = Optional[Union[None, Callable[[list[str]], str]]]

SepType: TypeAlias = Optional[Union[str, None]]

WrapType: TypeAlias = Optional[bool]

CaseSensitiveType: TypeAlias = Optional[bool]

RemoveEmptyRowsType: TypeAlias = Optional[bool]

FlagsType: TypeAlias = Optional[Union[numbers.Integral, None]]







