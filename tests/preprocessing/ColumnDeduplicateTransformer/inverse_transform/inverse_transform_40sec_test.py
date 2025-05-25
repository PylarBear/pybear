# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

import numpy as np
import pandas as pd
import polars as pl
import scipy.sparse as ss

from pybear.preprocessing._ColumnDeduplicateTransformer._inverse_transform. \
    _inverse_transform import _inverse_transform

from pybear.preprocessing import ColumnDeduplicateTransformer

from pybear.preprocessing._ColumnDeduplicateTransformer._partial_fit. \
    _parallel_column_comparer import _parallel_column_comparer

from pybear.utilities._nan_masking import nan_mask



@pytest.mark.parametrize('_dtype', ('flt', 'str'), scope='module')
@pytest.mark.parametrize('_has_nan', (True, False), scope='module')
@pytest.mark.parametrize('_keep', ('first', 'last', 'random'), scope='module')
@pytest.mark.parametrize('_do_not_drop', ([0,4,8], [3,7], [6,9]), scope='module')
@pytest.mark.parametrize('_equal_nan', (True, False), scope='module')
class TestInverseTransform:

    # verify that inverse_transform takes transformed back to original.
    # build an X with duplicates, use CDT to take out the duplicates under
    # different parameters (CDT transform() should be independently
    # validated), use inverse_transform to reconstruct back to the
    # original X.


    @pytest.mark.parametrize('_format',
        (
            'csr_matrix', 'coo_matrix', 'dia_matrix', 'lil_matrix', 'dok_matrix',
            'bsr_matrix', 'csr_array', 'coo_array', 'dia_array', 'lil_array',
            'dok_array', 'bsr_array'
        )
    )
    def test_rejects_all_ss_that_are_not_csc(
        self, _X_factory, _format, _keep, _do_not_drop, _equal_nan, _dtype,
        _has_nan, _shape
    ):

        # everything except ndarray, pd dataframe, & scipy csc matrix/array
        # are blocked. should raise.

        if _dtype == 'str':
            pytest.skip(reason=f"scipy sparse cannot take strings")

        # build X ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
        _X_wip = _X_factory(
            _dupl=[[0, 9], [2, 4, 7]],
            _has_nan=_has_nan,
            _format=_format,
            _dtype=_dtype,
            _shape=_shape
        )
        # END build X ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


        # _CDT is only being used here to get the legit TRFM_X. only test
        # the core _inverse_transform module, not _CDT.inverse_transform.
        _CDT = ColumnDeduplicateTransformer(
            keep=_keep,
            do_not_drop=_do_not_drop,
            conflict='ignore',
            rtol=1e-5,
            atol=1e-8,
            equal_nan=_equal_nan,
            n_jobs=1  # leave set at 1 because of confliction
        )

        TRFM_X = _CDT.fit_transform(_X_wip)

        with pytest.raises(AssertionError):
            _inverse_transform(
                X=TRFM_X,
                _removed_columns=_CDT.removed_columns_,
                _feature_names_in=None
            )


    @pytest.mark.parametrize('_format',
        ('np', 'pd', 'pl', 'csc_matrix', 'csc_array')
    )
    def test_accuracy(
        self, _X_factory, _format, _keep, _do_not_drop, _equal_nan, _dtype,
        _has_nan, _shape, _columns
    ):

        # Methodology: transform data, then transform back using
        # inverse_transform. the inverse transform must be equal to the
        # originally fitted data, except for nans. inverse transform
        # cannot infer the presence of nans in the original data.

        # everything except ndarray, pd dataframe, & scipy csc matrix/array
        # are blocked.

        # skip impossible conditions -- -- -- -- -- -- -- -- -- -- -- -- -- --
        if _dtype == 'str' and _format not in ('np', 'pd'):
            pytest.skip(reason=f"scipy sparse cannot take strings")
        # END skip impossible conditions -- -- -- -- -- -- -- -- -- -- -- -- --

        # build X ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
        _X_wip = _X_factory(
            _dupl=[[0, 9], [2, 4, 7]],
            _has_nan=_has_nan,
            _format=_format,
            _dtype=_dtype,
            _shape=_shape
        )

        if _format == 'np':
            _base_X = _X_wip.copy()
        elif _format in ['pd', 'pl']:
            _base_X = _X_wip.to_numpy()
        elif hasattr(_X_wip, 'toarray'):
            _base_X = _X_wip.toarray()
        else:
            raise Exception

        _base_X[nan_mask(_base_X)] = np.nan

        # END build X ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        # _CDT is only being used here to get the legit TRFM_X. only test
        # the core _inverse_transform module, not _CDT.inverse_transform.
        _CDT = ColumnDeduplicateTransformer(
            keep=_keep,
            do_not_drop=_do_not_drop,
            conflict='ignore',
            rtol=1e-5,
            atol=1e-8,
            equal_nan=_equal_nan,
            n_jobs=1  # leave set at 1 because of confliction
        )

        # fit v v v v v v v v v v v v v v v v v v v v
        _CDT.fit(_X_wip)
        # fit ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^

        # transform v v v v v v v v v v v v v v v v v v
        _dedupl_X = _CDT.transform(_X_wip, copy=True)
        # transform ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^


        # inverse transform v v v v v v v v v v v v v v v
        out = _inverse_transform(
            X=_dedupl_X,
            _removed_columns=_CDT.removed_columns_,
            _feature_names_in=_columns
        )
        # inverse transform ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^


        assert type(out) is type(_X_wip)

        assert out.shape == _base_X.shape


        # nans in string columns are being a real pain
        # _parallel_column_comparer instead of np.array_equal
        # _parallel_column_comparer cant do entire arrays at one time,
        # need to compare column by column

        for _og_idx in range(_shape[1]):

            _og_col = _base_X[:, [_og_idx]]

            if isinstance(_X_wip, np.ndarray):
                _out_col = out[:, [_og_idx]]
            elif isinstance(_X_wip, pd.core.frame.DataFrame):
                _out_col = out.iloc[:, [_og_idx]].to_numpy()
            elif isinstance(_X_wip, pl.DataFrame):
                # Polars uses zero-copy conversion when possible, meaning the
                # underlying memory is still controlled by Polars and marked
                # as read-only. NumPy and Pandas may inherit this read-only
                # flag, preventing modifications.
                # THE ORDER IS IMPORTANT HERE. CONVERT TO PANDAS FIRST, THEN COPY.
                _out_col = out.to_pandas().to_numpy()[:, [_og_idx]]
            elif hasattr(_X_wip, 'toarray'):
                _out_col = out[:, [_og_idx]].toarray()
            else:
                raise Exception

            # pizza
            _out_col[nan_mask(_out_col)] = np.nan
            _og_col[nan_mask(_og_col)] = np.nan

            assert _parallel_column_comparer(
                _out_col, _og_col, 1e-5, 1e-8, _equal_nan=True
            )[0]





