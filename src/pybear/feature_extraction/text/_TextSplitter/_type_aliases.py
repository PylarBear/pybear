# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Sequence
from typing_extensions import TypeAlias, Union

import re
import numbers


XContainer: TypeAlias = Sequence[str]

regexp: TypeAlias = Union[str, re.Pattern, Sequence[Union[str, re.Pattern]], None]

sep: TypeAlias = Union[str, set[str], None, list[Union[str, set[str], None]]]

maxsplit: TypeAlias = Union[numbers.Integral, Sequence[numbers.Integral], None]




