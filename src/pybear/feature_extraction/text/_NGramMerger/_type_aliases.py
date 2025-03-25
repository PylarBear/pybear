# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Callable, Sequence, Optional
from typing_extensions import TypeAlias, Union
import numpy.typing as npt

import re

import pandas as pd
import polars as pl




PythonTypes: TypeAlias = Sequence[Sequence[str]]

NumpyTypes: TypeAlias = npt.NDArray[str]

PandasTypes: TypeAlias = pd.DataFrame

PolarsTypes: TypeAlias = pl.DataFrame

XContainer: TypeAlias = Union[PythonTypes, NumpyTypes, PandasTypes, PolarsTypes]

NGramsType: TypeAlias = Sequence[Sequence[Union[str, re.Pattern]]]

CallableType: TypeAlias = Optional[Callable[[Sequence[str]], str]]

SepType: TypeAlias = Optional[str]










