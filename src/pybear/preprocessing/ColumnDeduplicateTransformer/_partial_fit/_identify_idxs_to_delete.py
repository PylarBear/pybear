# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from .._type_aliases import DoNotDropType
from typing import Literal
from typing_extensions import Union

import numpy as np




def _identify_idxs_to_delete(
    _duplicates: list[list[int]],
    _do_not_drop: DoNotDropType,
    _conflict: Union[Literal['raise'], Literal['ignore']]
) -> list[int]:


    # apply the keep, do_not_drop, and conflict rules to the returned idxs

    _n_do_not_drop = \
        lambda _set: len(set(self._do_not_drop).intersection(_set))

    _removed_columns = []
    for _set in self.duplicates_:
        _n = _n_do_not_drop(_set)
        if _n == 1:
        # pizza

        elif self._keep == 'first':

            _removed_columns.append(_set[0])
        elif self._keep == 'last':
            _removed_columns.append(_set[-1])
        elif self._keep == 'random':
            _removed_columns.append(np.random.choice(_set))

    # look for conflicts in d













