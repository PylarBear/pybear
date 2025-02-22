# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Sequence
from typing_extensions import TypeAlias, Union





XContainer: TypeAlias = Union[Sequence[str], Sequence[Sequence[str]]]

sep: Union[str, Sequence[str]]

fill: str









