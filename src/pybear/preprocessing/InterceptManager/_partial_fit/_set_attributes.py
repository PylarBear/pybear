# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Iterable
from typing_extensions import Union

import numpy as np



def _set_attributes(
    constant_columns_: dict[int, any],
    _instructions: dict[str, Union[None, Iterable[int]]],
    _n_features: int
) -> tuple[dict[int, any], dict[int, any]]:


    kept_columns_: dict[int, any] = {}
    removed_columns_: dict[int, any] = {}
    _column_mask = np.ones(len(_n_features)).astype(bool)

    for col_idx, constant_value in constant_columns_.items():
        if col_idx in _instructions['keep'] or {}:
            kept_columns_[col_idx] = constant_value
        elif col_idx in _instructions['add'] or {}:
            kept_columns_[col_idx] = constant_value
        elif col_idx in _instructions['delete'] or {}:
            removed_columns_[col_idx] = constant_value
            _column_mask[col_idx] = False



    self._column_mask


    return self.kept_columns_, self.removed_columns_



