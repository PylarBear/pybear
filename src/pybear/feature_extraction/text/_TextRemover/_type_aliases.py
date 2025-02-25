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



XContainer: TypeAlias = Union[Sequence[str], Sequence[Sequence[str]]]

StrType: TypeAlias = Union[str, set[str]]
StrRemoveType: TypeAlias = \
    Union[None, StrType, list[Union[StrType, Literal[False]]]]

RegExpType: TypeAlias = Union[str, re.Pattern]
RegExpRemoveType: TypeAlias = \
    Union[None, RegExpType, list[Union[RegExpType, Literal[False]]]]

FlagType: TypeAlias = Union[None, numbers.Integral]
RegExpFlagsType: TypeAlias = \
    Union[FlagType, list[Union[FlagType, Literal[False]]]]

RowSupportType: TypeAlias = npt.NDArray[bool]




