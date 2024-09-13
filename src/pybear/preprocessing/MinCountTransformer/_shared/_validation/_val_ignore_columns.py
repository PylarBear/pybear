# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from typing_extensions import Union

import numpy as np

from ._core_ign_cols_handle_as_bool import _core_ign_cols_handle_as_bool

from ..._type_aliases import IgnColsHandleAsBoolDtype



def _val_ignore_columns(
        kwarg_value: IgnColsHandleAsBoolDtype,
        _mct_has_been_fit: bool = False,
        _n_features_in: Union[None, int] = None,
        _feature_names_in: Union[None, np.ndarray[str]] = None,
    ) -> Union[callable, np.ndarray[int], np.ndarray[str]]:


    _ignore_columns = _core_ign_cols_handle_as_bool(
        kwarg_value,
        'ignore_columns',
        _mct_has_been_fit=_mct_has_been_fit,
        _n_features_in=_n_features_in,
        _feature_names_in=_feature_names_in
    )

    return _ignore_columns







