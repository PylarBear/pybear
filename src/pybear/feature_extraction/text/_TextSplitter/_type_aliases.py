# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Literal, Sequence
from typing_extensions import TypeAlias, Union

import re
import numbers



XContainer: TypeAlias = Sequence[str]

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

RegExpType: TypeAlias = Union[numbers.Integral, None]
RegExpFlagsType: TypeAlias = \
    Union[RegExpType, list[Union[RegExpType, Literal[False]]]]






















