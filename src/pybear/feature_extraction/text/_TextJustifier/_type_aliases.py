# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import (
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



PythonTypes: TypeAlias = Union[Sequence[str], Sequence[Sequence[str]], set[str]]

NumpyTypes: TypeAlias = npt.NDArray

PandasTypes: TypeAlias = Union[pd.Series, pd.DataFrame]

PolarsTypes: TypeAlias = Union[pl.Series, pl.DataFrame]

XContainer: TypeAlias = Union[PythonTypes, NumpyTypes, PandasTypes, PolarsTypes]

XWipContainer: TypeAlias = Union[list[str], list[list[str]]]

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

NCharsType: TypeAlias = Optional[numbers.Integral]

CoreSepBreakType: TypeAlias = \
    Union[str, Sequence[str], re.Pattern[str], Sequence[re.Pattern[str]]]

SepType: TypeAlias = Optional[CoreSepBreakType]

LineBreakType: TypeAlias = Optional[Union[None, CoreSepBreakType]]

CoreSepBreakWipType: TypeAlias = Union[re.Pattern[str], tuple[re.Pattern[str], ...]]

SepWipType: TypeAlias = CoreSepBreakWipType

LineBreakWipType: TypeAlias = Union[None, CoreSepBreakWipType]

CaseSensitiveType: TypeAlias = Optional[bool]

SepFlagsType: TypeAlias = Optional[Union[numbers.Integral, None]]

LineBreakFlagsType: TypeAlias = Optional[Union[numbers.Integral, None]]

BackfillSepType: TypeAlias = Optional[str]

Join2DType: TypeAlias = Optional[str]








