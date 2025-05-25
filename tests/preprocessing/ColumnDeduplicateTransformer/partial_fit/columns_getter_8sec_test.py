# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

import numpy as np
import scipy.sparse as ss

from pybear.preprocessing._ColumnDeduplicateTransformer._partial_fit. \
    _columns_getter import _columns_getter

from pybear.utilities._nan_masking import nan_mask



class TestColumnsGetter:


    @pytest.mark.parametrize('_col_idxs',
        (0, 20, (0,), (20,), (0,1), (0,20), (100,200))
    )
    def test_rejects_idx_out_of_col_range(self, _X_factory, _col_idxs, _shape):

        _out_of_range = False
        try:
            # is a tuple
            tuple(_col_idxs)
            for _idx in _col_idxs:
                if _idx not in range(_shape[1]):
                    _out_of_range = True
        except:
            # is int
            if _col_idxs not in range(_shape[1]):
                _out_of_range = True

        _X = _X_factory(
            _format='np',
            _dtype='flt',
            _has_nan=False,
            _shape=_shape
        )

        if _out_of_range:
            with pytest.raises(AssertionError):
                _columns_getter(_X, _col_idxs)
        else:
            assert isinstance(_columns_getter(_X, _col_idxs), np.ndarray)


    @pytest.mark.parametrize('_dtype', ('flt', 'str'))
    @pytest.mark.parametrize('_format',
        (
        'np', 'pd', 'pl', 'csr_array', 'csr_matrix', 'csc_array', 'csc_matrix',
        'coo_array', 'coo_matrix', 'dia_array', 'dia_matrix', 'lil_array',
        'lil_matrix', 'dok_array', 'dok_matrix', 'bsr_array', 'bsr_matrix'
        )
    )
    @pytest.mark.parametrize('_col_idxs',
        (0, 1, 2, (0,), (1,), (2,), (0,1), (0,2), (1,2), (0,1,2))
    )
    @pytest.mark.parametrize('_has_nan', (True, False), scope='module')
    def test_accuracy(
        self, _X_factory, _has_nan, _col_idxs, _format, _dtype, _columns,
        _shape
    ):

        # coo, dia, & bsr matrix/array are blocked. should raise here.

        if _dtype == 'str' and _format not in ('np', 'pd', 'pl'):
            pytest.skip(reason=f"scipy sparse cant take non numeric data")

        # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

        # _columns_getter only allows ss csc. everything else should raise.

        _X_wip = _X_factory(
            _dupl=None,
            _format=_format,
            _dtype=_dtype,
            _has_nan=_has_nan,
            _columns=None,
            _shape=_shape
        )

        # use _X_ref to referee whether _columns_getter pulled the
        # correct columns from _X_wip
        if _format == 'np':
            _X_ref = _X_wip.copy()
        elif _format in ['pd', 'pl']:
            _X_ref = _X_wip.to_numpy()
        elif hasattr(_X_wip, 'toarray'):
            _X_ref = _X_wip.toarray()
        else:
            raise Exception


        # need to get rid of junky nans put in pd by X_factory,
        # array_equal cant handle them. verified 25_05_24
        _X_ref[nan_mask(_X_ref)] = np.nan

        # ensure _col_idxs, when tuple, is sorted, _columns_getter does this,
        # make sure any careless changes made to this test are handled.
        try:
            iter(_col_idxs)
            _col_idxs = tuple(sorted(list(_col_idxs)))
        except:
            pass

        # pass _col_idxs as given (int or tuple) to _columns getter
        # verify _columns_getter rejects coo, dia, and bsr
        if hasattr(_X_wip, 'toarray') \
                and not isinstance(_X_wip, (ss.csc_matrix, ss.csc_array)):
            # this is raised by top-of-file validation
            with pytest.raises(AssertionError):
                _columns_getter(_X_wip, _col_idxs)
            pytest.skip(reason=f"cant do anymore tests after exception")
        else:
            _columns = _columns_getter(_X_wip, _col_idxs)


        # assertions v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v

        assert isinstance(_columns, np.ndarray)
        assert len(_columns.shape) == 2

        # now that _columns getter has seen the given _col_idxs, convert all
        # given _col_idxs to tuple to make _X[:, _col_idxs] slice right, below
        try:
            len(_col_idxs)  # except if is integer
        except:  # if is integer change to tuple
            _col_idxs = (_col_idxs,)

        assert _columns.shape[1] == len(_col_idxs)


        # 25_05_24 pd numeric with junky nan-likes are coming out of
        # _columns_getter as dtype object. since _columns_getter produces
        # an intermediary container that is used to find constants and
        # doesnt impact the container coming out of transform, ok to let
        # that condition persist and just fudge the dtype for this test.
        if _dtype == 'flt':
            assert np.array_equal(
                _columns.astype(np.float64),
                _X_ref[:, _col_idxs].astype(np.float64),
                equal_nan=True
            )
        elif _dtype == 'str':
            assert np.array_equal(
                _columns.astype(str), _X_ref[:, _col_idxs].astype(str)
            )




