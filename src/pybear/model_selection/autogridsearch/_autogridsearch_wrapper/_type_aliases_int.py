# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Sequence
from typing_extensions import TypeAlias

import numbers



# see _type_aliases - int subtypes for DataType, GridType
IntDataType: TypeAlias = numbers.Integral
IntGridType: TypeAlias = Sequence[IntDataType]





