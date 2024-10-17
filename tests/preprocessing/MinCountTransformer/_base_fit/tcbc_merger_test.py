# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from pybear.preprocessing.MinCountTransformer._base_fit._tcbc_merger import \
    _tcbc_merger

from pybear.utilities._nan_masking import nan_mask

import numpy as np

import pytest



class TestTCBCMergerTest:


    @staticmethod
    @pytest.fixture(scope='module')
    def _tcbc():
        return {
            0: {0: 10, 1: 5, 2: 8, 3: 15, np.nan: 3},
            1: {0: 5, 1: 7, 2: 3, 3: 2, 'nan': 5}
        }


    @staticmethod
    @pytest.fixture(scope='module')
    def _tcbc_no_nan():
        return {
            0: {0: 10, 1: 5, 2: 8, 3: 15},
            1: {0: 5, 1: 7, 2: 3, 3: 2}
        }


    def test_correctly_doubles_values(self, _tcbc_no_nan):
        # when no nans

        out = _tcbc_merger(
            _DTYPE_UNQS_CTS_TUPLES=[('int', v) for k,v in _tcbc_no_nan.items()],
            _tcbc=_tcbc_no_nan
        )

        assert np.array_equal(list(out.keys()), list(_tcbc_no_nan.keys()))

        for _c_idx in out:
            assert np.array_equal(
                list(out[_c_idx].keys()),
                list(_tcbc_no_nan[_c_idx].keys())
            )
            for _unq in out[_c_idx]:
                assert out[_c_idx][_unq] == 2 * _tcbc_no_nan[_c_idx][_unq]





    def test_correctly_adds_nans(self, _tcbc):

        out = _tcbc_merger(
            _DTYPE_UNQS_CTS_TUPLES=[('int', v) for k,v in _tcbc.items()],
            _tcbc=_tcbc
        )


        assert np.array_equal(list(out.keys()), list(_tcbc.keys()))

        for _c_idx in out:
            _out_keys = np.fromiter(out[_c_idx], dtype=object)
            _tcbc_keys = np.fromiter(out[_c_idx], dtype=object)
            assert np.array_equal(
                _out_keys[np.logical_not(nan_mask(_out_keys))],
                _tcbc_keys[np.logical_not(nan_mask(_tcbc_keys))]
            )
            for _unq in _out_keys:
                assert out[_c_idx][_unq] == 2 * _tcbc[_c_idx][_unq]



















