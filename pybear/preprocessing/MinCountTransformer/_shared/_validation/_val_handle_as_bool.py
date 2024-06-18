# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

from typing import Union
from ..._type_aliases import OriginalDtypesDtype

import numpy as np

from ._core_ign_cols_handle_as_bool import _core_ign_cols_handle_as_bool
from ._val_original_dtypes import _val_original_dtypes

from ..._type_aliases import IgnColsHandleAsBoolDtype



def _val_handle_as_bool(
        kwarg_value: IgnColsHandleAsBoolDtype,
        _mct_has_been_fit: bool = False,
        _n_features_in: Union[None, int] = None,
        _feature_names_in: Union[None, np.ndarray[str]] = None,
        _original_dtypes: Union[None, OriginalDtypesDtype] = None,
    ) -> Union[callable, np.ndarray[int], np.ndarray[str]]:



    _handle_as_bool = _core_ign_cols_handle_as_bool(
        kwarg_value,
        'handle_as_bool',
        _mct_has_been_fit=_mct_has_been_fit,
        _n_features_in=_n_features_in,
        _feature_names_in=_feature_names_in
    )


    if _mct_has_been_fit:   # (_original_dtypes and _n_features_in not None)

        _original_dtypes = _val_original_dtypes(_original_dtypes)

        # kwarg_value should have been converted to indices by _cichab if it
        # wasnt a callable
        if not callable(_handle_as_bool) and \
                'obj' in _original_dtypes[_handle_as_bool]:
            MASK = (_original_dtypes[_handle_as_bool] == 'obj')
            IDXS = ', '.join(map(str, _handle_as_bool[MASK]))
            raise ValueError(f"cannot use handle_as_bool on str/object columns "
                f"--- column index(es) == {IDXS}")


    return _handle_as_bool








