# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from pybear.preprocessing.ColumnDeduplicateTransformer._inverse_transform. \
    _inverse_transform import _inverse_transform

from pybear.preprocessing import ColumnDeduplicateTransformer

from pybear.preprocessing.ColumnDeduplicateTransformer._partial_fit. \
    _parallel_column_comparer import _parallel_column_comparer

import numpy as np
import pandas as pd
import scipy.sparse as ss

import pytest






@pytest.mark.parametrize('_dtype', ('flt', 'str'), scope='module')
@pytest.mark.parametrize('_has_nan', (True, False), scope='module')
@pytest.mark.parametrize('_keep', ('first', 'last', 'random'), scope='module')
@pytest.mark.parametrize('_do_not_drop', ([0,4,8], [3,7], [6,9]), scope='module')
@pytest.mark.parametrize('_equal_nan', (True, False), scope='module')
class TestInverseTransform:

    # assert that inverse_transform takes transformed back to original
    # build an X with duplicates, use CDT to take out the duplicates under
    # different parameters (CDT transform() should be independently
    # validated), use inverse_transform to reconstruct back to the
    # original X.

    # fixtures ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    @staticmethod
    @pytest.fixture(scope='module')
    def _shape():
        return (20, 10)


    @staticmethod
    @pytest.fixture(scope='module')
    def _rtol_atol():
        return (1e-5, 1e-8)


    @staticmethod
    @pytest.fixture(scope='module')
    def _dupl():
        return [
            [0, 9],
            [2, 4, 7]
        ]


    @staticmethod
    @pytest.fixture(scope='module')
    def _columns(_master_columns, _shape):
        return _master_columns.copy()[:_shape[1]]


    @staticmethod
    @pytest.fixture(scope='module')
    def _dupl_X(_X_factory, _shape, _dupl, _dtype, _has_nan):

        return _X_factory(
            _dupl=_dupl,
            _has_nan=_has_nan,
            _format='np',
            _dtype=_dtype,
            _shape=_shape
        )

    # END fixtures ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *




    @pytest.mark.parametrize('_format',
        (
            'ndarray', 'df', 'csr_matrix', 'csc_matrix', 'coo_matrix',
            'dia_matrix', 'lil_matrix', 'dok_matrix', 'bsr_matrix',
            'csr_array', 'csc_array', 'coo_array', 'dia_array', 'lil_array',
            'dok_array', 'bsr_array'
        )
    )
    def test_accuracy(
        self, _dupl_X, _format, _keep, _do_not_drop, _equal_nan, _dtype,
        _has_nan, _shape, _columns, _rtol_atol
    ):

        # everything except ndarray, pd dataframe, & scipy csc matrix/array
        # are blocked. should raise.

        if _dtype == 'str' and _format not in ('ndarray', 'df'):
            pytest.skip(reason=f"scipy sparse cannot take strings")

        _base_X = _dupl_X.copy()

        _CDT = ColumnDeduplicateTransformer(
            keep=_keep,
            do_not_drop=_do_not_drop,
            conflict='ignore',
            # rtol: Optional[Real] = 1e-5,
            # atol: Optional[Real] = 1e-8,
            equal_nan=_equal_nan,
            n_jobs=1  # leave set at 1 because of confliction
        )

        # fit v v v v v v v v v v v v v v v v v v v v
        # fit on the numpy X
        # this actually isnt necessary, fit will take whatever
        _CDT.fit(_base_X)
        # fit ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^

        # transform v v v v v v v v v v v v v v v v v v
        # transform the numpy X
        # this actually isnt necessary, transform will take whatever
        _dedupl_X = _CDT.transform(_base_X, copy=True)
        # transform ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^

        # convert _dedupl_X to the test format

        if _format == 'ndarray':
            _X_wip = _dedupl_X
        elif _format == 'df':
            _X_wip = pd.DataFrame(
                data=_dedupl_X,
                columns=_columns[_CDT.column_mask_]
            )
        elif _format == 'csr_matrix':
            _X_wip = ss._csr.csr_matrix(_dedupl_X)
        elif _format == 'csc_matrix':
            _X_wip = ss._csc.csc_matrix(_dedupl_X)
        elif _format == 'coo_matrix':
            _X_wip = ss._coo.coo_matrix(_dedupl_X)
        elif _format == 'dia_matrix':
            _X_wip = ss._dia.dia_matrix(_dedupl_X)
        elif _format == 'lil_matrix':
            _X_wip = ss._lil.lil_matrix(_dedupl_X)
        elif _format == 'dok_matrix':
            _X_wip = ss._dok.dok_matrix(_dedupl_X)
        elif _format == 'bsr_matrix':
            _X_wip = ss._bsr.bsr_matrix(_dedupl_X)
        elif _format == 'csr_array':
            _X_wip = ss._csr.csr_array(_dedupl_X)
        elif _format == 'csc_array':
            _X_wip = ss._csc.csc_array(_dedupl_X)
        elif _format == 'coo_array':
            _X_wip = ss._coo.coo_array(_dedupl_X)
        elif _format == 'dia_array':
            _X_wip = ss._dia.dia_array(_dedupl_X)
        elif _format == 'lil_array':
            _X_wip = ss._lil.lil_array(_dedupl_X)
        elif _format == 'dok_array':
            _X_wip = ss._dok.dok_array(_dedupl_X)
        elif _format == 'bsr_array':
            _X_wip = ss._bsr.bsr_array(_dedupl_X)
        else:
            raise Exception


        # inverse transform v v v v v v v v v v v v v v v
        if _format in ('ndarray', 'df', 'csc_matrix', 'csc_array'):
            out = _inverse_transform(
                X=_X_wip,
                _removed_columns=_CDT.removed_columns_,
                _feature_names_in=_columns
            )
        else:
            with pytest.raises(AssertionError):
                _inverse_transform(
                    X=_X_wip,
                    _removed_columns=_CDT.removed_columns_,
                    _feature_names_in=_columns
                )
            pytest.skip(reason=f"cant do more tests after exception")
        # inverse transform ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^


        assert type(out) is type(_X_wip)

        assert out.shape == _base_X.shape


        # nans in string columns are being a real pain
        # _parallel_column_comparer instead of np.array_equal
        # _parallel_column_comparer cant do entire arrays at one time,
        # need to compare column by column

        for _idx in range(_X_wip.shape[1]):

            _og_col = _base_X[:, _idx]

            if isinstance(_X_wip, np.ndarray):
                _out_col = out[:, _idx]
            elif isinstance(_X_wip, pd.core.frame.DataFrame):
                _out_col = out.iloc[:, _idx]
            elif isinstance(_X_wip, (ss.csc_matrix, ss.csc_array)):
                _out_col = out[:, [_idx]].toarray().ravel()
            else:
                raise Exception

            assert _parallel_column_comparer(
                _out_col, _og_col, *_rtol_atol, _equal_nan=True
            )















