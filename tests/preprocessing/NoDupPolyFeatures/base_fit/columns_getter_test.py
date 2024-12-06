# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.preprocessing.NoDupPolyFeatures._base_fit._columns_getter import (
    _columns_getter
)

import numpy as np
import pandas as pd
import scipy.sparse as ss

import pytest




# this mark needs to stay here because fixtures _X_num & _X_str need it
@pytest.mark.parametrize('_has_nan', (False,), scope='module')   # (True, False) pizza
class TestColumnGetter:

    # fixtures ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

    @staticmethod
    @pytest.fixture(scope='module')
    def _shape():
        return (10, 5)


    @staticmethod
    @pytest.fixture(scope='module')
    def _X_num(_X_factory, _shape, _has_nan):

        return _X_factory(
            _dupl=None,
            _format='np',
            _dtype='flt',
            _has_nan=_has_nan,
            _columns=None,
            _shape=_shape
        )


    @staticmethod
    @pytest.fixture(scope='module')
    def _X_str(_X_factory, _shape, _has_nan):

        return _X_factory(
            _dupl=None,
            _format='np',
            _dtype='str',
            _has_nan=_has_nan,
            _columns=None,
            _shape=_shape
        )

    # END fixtures ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **



    @pytest.mark.parametrize('_col_idxs',
        (0, 20, (0,), (20,), (0,1), (0,20), (100,200))
    )
    def test_rejects_idx_out_of_col_range(
        self, _X_factory, _has_nan, _col_idxs, _shape):

        _X_wip = _X_factory(
            _dupl=None,
            _format='np',
            _dtype='flt',
            _has_nan=_has_nan,
            _columns=None,
            _shape=_shape
        )

        _out_of_range = False
        try:
            tuple(_col_idxs)
            # is a tuple
            for _idx in _col_idxs:
                if _idx not in range(_shape[1]):
                    _out_of_range = True
        except:
            # is int
            if _col_idxs not in range(_shape[1]):
                _out_of_range = True


        if _out_of_range:
            with pytest.raises(AssertionError):
                _columns_getter(_X_wip, _col_idxs)
        else:
            _columns = _columns_getter(_X_wip, _col_idxs)



    @pytest.mark.parametrize('_dtype', ('num', 'str'))
    @pytest.mark.parametrize('_format',
        (
        'ndarray', 'df', 'csr_matrix', 'csc_matrix', 'coo_matrix', 'dia_matrix',
        'lil_matrix', 'dok_matrix', 'bsr_matrix', 'csr_array', 'csc_array',
        'coo_array', 'dia_array', 'lil_array', 'dok_array', 'bsr_array',
        )
    )
    @pytest.mark.parametrize('_col_idxs',
        (0, 1, 2, (0,), (1,), (2,), (0,1), (0,2), (1,2), (0,1,2))
    )
    def test_accuracy(
        self, _has_nan, _dtype, _format, _col_idxs, _shape, _X_num, _X_str,
        _master_columns
    ):

        if _dtype == 'str' and _format not in ('ndarray', 'df'):
            pytest.skip(reason=f"scipy sparse cant take non numeric data")

        if _dtype == 'num':
            _X = _X_num
        elif _dtype == 'str':
            _X = _X_str

        if _format == 'ndarray':
            _X_wip = _X
        elif _format == 'df':
            _X_wip = pd.DataFrame(
                data=_X,
                columns=_master_columns.copy()[:_shape[1]]
            )
        elif _format == 'csr_matrix':
            _X_wip = ss._csr.csr_matrix(_X)
        elif _format == 'csc_matrix':
            _X_wip = ss._csc.csc_matrix(_X)
        elif _format == 'coo_matrix':
            _X_wip = ss._coo.coo_matrix(_X)
        elif _format == 'dia_matrix':
            _X_wip = ss._dia.dia_matrix(_X)
        elif _format == 'lil_matrix':
            _X_wip = ss._lil.lil_matrix(_X)
        elif _format == 'dok_matrix':
            _X_wip = ss._dok.dok_matrix(_X)
        elif _format == 'bsr_matrix':
            _X_wip = ss._bsr.bsr_matrix(_X)
        elif _format == 'csr_array':
            _X_wip = ss._csr.csr_array(_X)
        elif _format == 'csc_array':
            _X_wip = ss._csc.csc_array(_X)
        elif _format == 'coo_array':
            _X_wip = ss._coo.coo_array(_X)
        elif _format == 'dia_array':
            _X_wip = ss._dia.dia_array(_X)
        elif _format == 'lil_array':
            _X_wip = ss._lil.lil_array(_X)
        elif _format == 'dok_array':
            _X_wip = ss._dok.dok_array(_X)
        elif _format == 'bsr_array':
            _X_wip = ss._bsr.bsr_array(_X)
        else:
            raise Exception

        try:
            iter(_col_idxs)
            _col_idxs = tuple(sorted(list(_col_idxs)))
        except:
            pass

        _columns = _columns_getter(_X_wip, _col_idxs)

        assert isinstance(_columns, np.ndarray)

        try:
            len(_col_idxs)  # except if is integer
        except: # if is integer
            # change int to tuple to make _X[:, _col_idxs] slice right, below
            _col_idxs = (_col_idxs,)



        if hasattr(_X_wip, 'toarray'):
            assert _columns.shape[1] == 1
        else:
            assert _columns.shape[1] == len(_col_idxs)

        # since all the various _X_wips came from _X, just use _X to referee
        # whether _columns_getter pulled the correct column from _X_wip

        if _dtype == 'num':
            if hasattr(_X_wip, 'toarray'):
                __ = ss.csc_array(_X[:, _col_idxs])
                _stack = np.hstack((__.indices, __.data)).reshape((-1, 1))
                assert np.array_equal(_columns, _stack, equal_nan=True)
            elif not hasattr(_X_wip, 'toarray'):
                assert np.array_equal(_columns, _X[:, _col_idxs], equal_nan=True)
        elif _dtype == 'str':
            assert np.array_equal(
                _columns.astype(str),
                _X[:, _col_idxs].astype(str)
            )













