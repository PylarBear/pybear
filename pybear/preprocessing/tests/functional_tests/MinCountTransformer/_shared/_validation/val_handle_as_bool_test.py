# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

import pytest
import numpy as np

from pybear.preprocessing.MinCountTransformer._shared._validation. \
    _val_handle_as_bool import _val_handle_as_bool



class TestValHandleAsBool:

    # kwarg_value: IgnColsHandleAsBoolDtype,
    # _mct_has_been_fit: bool = False, # pass hasattr(self, 'n_features_in_')
    # _n_features_in: Union[None, int] = None,
    # _feature_names_in: Union[None, np.ndarray[str]] = None,
    # _original_dtypes: Union[None, OriginalDtypesDtype] = None,
    # ) -> Union[callable, np.ndarray[int], np.ndarray[str]]:


    # core tests done in core_ign_columns_handle_as_bool_test

    @pytest.mark.parametrize('og_dtype', ('int', 'obj', 'float'))
    def test_rejects_h_a_b_on_obj_columns(self, og_dtype):

        _DTYPES = np.array(['int', og_dtype, 'int', 'float'])

        if og_dtype == 'obj':
            with pytest.raises(ValueError):
                _val_handle_as_bool(
                    np.array([0 ,1, 2], dtype=np.uint32),
                    _mct_has_been_fit=True,
                    _n_features_in=4,
                    _feature_names_in=None,
                    _original_dtypes=_DTYPES
                )
        else:
            out = _val_handle_as_bool(
                [0 ,1, 2],
                _mct_has_been_fit=True,
                _n_features_in=4,
                _feature_names_in=None,
                _original_dtypes=_DTYPES
            )

            assert isinstance(out, np.ndarray)
            assert np.array_equiv(out, [0, 1, 2])





















