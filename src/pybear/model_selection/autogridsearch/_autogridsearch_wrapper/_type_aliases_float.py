# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Sequence
from typing_extensions import TypeAlias

import numbers



# see _type_aliases; float subtypes for DataType & GridType
FloatDataType: TypeAlias = numbers.Real
FloatGridType: TypeAlias = Sequence[FloatDataType]





