# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pytest

import numpy as np
import pandas as pd  # pizza
from uuid import uuid4
import scipy.sparse as ss

from pybear.preprocessing.NoDupPolyFeatures._base_fit._constant_handling. \
    _get_constant_columns import _get_constant_columns


pytest.skip(reason=f"pizza not finished", allow_module_level=True)


class TestGetConstantColumns:

    # fixtures v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^
    @staticmethod
    @pytest.fixture(scope='module')
    def _shape():
        return (10,3)


    @staticmethod
    @pytest.fixture(scope='module')
    def _constant_idxs(_shape: tuple[int]):
        return [1, _shape[1]-1]


    # @staticmethod
    # @pytest.fixture(scope='module')
    # def _good_np(_shape: tuple[int], _constant_idxs:list[int]):
    #
    #     _ = np.random.randint(0, 10, _shape)
    #     for idx in _constant_idxs:
    #         _[:, idx] = 0
    #
    #     return _
    #
    # @staticmethod
    # @pytest.fixture(scope='module')
    # def _good_pd(_good_np):
    #
    #     columns = [str(uuid4())[:4] for _ in range(_shape[1])]
    #
    #     return pd.DataFrame(data=_good_np, columns=columns)
    #
    #
    # @staticmethod
    # @pytest.fixture(scope='module')
    # def _good_ss(_good_np):
    #
    #     return ss.csc_array(_good_np)

    # END fixtures v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v


    # test validation ^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v
    @pytest.mark.parametrize('junk_X',
        (-3.14,-1,0,1,3.14,True,False,None,'junk',[0,1],(0,1),{'a':1}, lambda x: x)
    )
    def test_rejects_junk_X(self, junk_X):
        with pytest.raises(AssertionError):
            _get_constant_columns(junk_X, True)


    def test_accepts_np_pd_ss_X(self, _X_factory):
        _get_constant_columns(_X_factory(_format='np'), True)
        _get_constant_columns(_X_factory(_format='pd'), False)
        _get_constant_columns(_X_factory(_format='csc'), True)


    @pytest.mark.parametrize('_junk',
        (-3.14,-1,0,1,3.14,None,'junk',[0,1],(0,1),{'a':1}, lambda x: x)
    )
    def test_rejects_non_bool_as_indices(self, _X_factory, _junk):
        with pytest.raises(AssertionError):
            _get_constant_columns(_X_factory(_format='np'), _junk)


    @pytest.mark.parametrize('_bool', (True, False))
    def test_accepts_bool_as_indices(self, _X_factory, _bool):
        _get_constant_columns(_X_factory(_format='np'), _bool)

    # END test validation ^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^

    @pytest.mark.parametrize('_X_format', ('np', 'pd', 'csc', 'csr', 'coo'))
    @pytest.mark.parametrize('_X_dtype', ('flt', 'int'))
    @pytest.mark.parametrize('_as_indices', (True, False))
    @pytest.mark.parametrize('_has_consts', (True, False))
    @pytest.mark.parametrize('_has_nan', (True, False))
    def test_accuracy(
        self, _X_format, _X_dtype, _as_indices, _has_consts, _has_nan,
        _X_factory, _constant_idxs, _shape
    ):

        _wip_X = _X_factory(
            _dupl=None,
            _has_nan=_has_nan,
            _format=_X_format,
            _dtype=_X_dtype,
            _columns=[str(uuid4())[:4] for _ in range(_shape[1])],
            _constants=_constant_idxs if _has_consts else None,
            _zeros=None,
            _shape=_shape
        )

        # pizza
        # if _has_consts:
        #     if _X_format == 'np':
        #         _wip_X[:, _constant_idxs] = 0
        #     elif _X_format == 'pd':
        #         _wip_X.iloc[:, _constant_idxs] = 0
        #     else:
        #         _og_ss_type = type(_wip_X)
        #         # must convert to csc, csr, whatever... coo cant take assignment
        #         _wip_X = _wip_X.tocsc()
        #         _wip_X[:, _constant_idxs] = 0
        #         _wip_X = _og_ss_type(_wip_X)
        #
        #         # check that ss _wip_X was built correctly
        #         _check_X = _wip_X.toarray()
        #         assert not np.any(_check_X[:, [_constant_idxs[0]]])
        #         assert not np.any(_check_X[:, [_constant_idxs[1]]])

        out = _get_constant_columns(_wip_X, _equal_nan, _rtol, _atol, _as_indices)

        assert isinstance(out, np.ndarray)

        if not _has_consts:
            if _as_indices:
                assert out.dtype == np.int32
                assert len(out) == 0

            elif not _as_indices:
                assert out.dtype == bool
                assert len(out) == _shape[1]

        elif _has_consts:
            if _as_indices:
                assert out.dtype == np.int32
                assert np.array_equal(out, _constant_idxs)

            elif not _as_indices:
                assert out.dtype == bool
                assert len(out) == _shape[1]
                for idx, value in enumerate(out):
                    if idx in _constant_idxs:
                        assert value is np.True_
                    else:
                        assert value is np.False_










