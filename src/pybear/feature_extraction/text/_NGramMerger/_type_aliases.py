# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Callable, Sequence
from typing_extensions import TypeAlias, Union
import numpy.typing as npt

import re

import pandas as pd
import polars as pl




PythonTypes: TypeAlias = Union[Sequence[str], Sequence[Sequence[str]]]

NumpyTypes: TypeAlias = npt.NDArray[str]

PandasTypes: TypeAlias = Union[pd.Series, pd.DataFrame]

PolarsTypes: TypeAlias = Union[pl.Series, pl.DataFrame]

XContainer: TypeAlias = Union[Sequence[str], Sequence[Sequence[str]]]


StrKeysType: TypeAlias = tuple[str, ...]
StrValuesType: TypeAlias = Union[None, str, Callable[[str, ...], str]]
StrNGramHandlerType: TypeAlias = dict[StrKeysType, StrValuesType]

RegExpKeysType: TypeAlias = tuple[Union[str, re.Pattern], ...]
RegExpValuesType: TypeAlias = Union[None, str, Callable[[str, ...], str]]
RegExpNGramHandlerType: TypeAlias = dict[RegExpKeysType, RegExpValuesType]


StrSepType: TypeAlias = Union[str, set[str]]
RegExpSepType: TypeAlias = Union[str, re.Pattern]










