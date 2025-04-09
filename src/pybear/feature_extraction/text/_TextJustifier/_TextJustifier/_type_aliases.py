# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Optional, Sequence
from typing_extensions import TypeAlias, Union


StrSepType: TypeAlias = Optional[Union[str, Sequence[str]]]

StrLineBreakType: TypeAlias = Optional[Union[str, Sequence[str], None]]

CaseSensitiveType: TypeAlias = Optional[bool]










