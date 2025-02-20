# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Callable, Sequence
from typing_extensions import TypeAlias, Union



XContainer: TypeAlias = Sequence[str]
WIPXContainer: TypeAlias = Union[Sequence[str], Sequence[Sequence[str]]]


MenuDictType: TypeAlias = dict[str, dict[str, Union[str, Callable]]]








