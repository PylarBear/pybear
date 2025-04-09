# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Optional, Sequence
from typing_extensions import TypeAlias, Union

import numbers
import re



RegExpSepType: TypeAlias = Optional[Union[re.Pattern[str], Sequence[re.Pattern[str]]]]

SepFlagsType: TypeAlias = Optional[Union[numbers.Integral, None]]

RegExpLineBreakType: TypeAlias = \
    Optional[Union[re.Pattern[str], Sequence[re.Pattern[str]], None]]

LineBreakFlagsType: TypeAlias = Optional[Union[numbers.Integral, None]]





