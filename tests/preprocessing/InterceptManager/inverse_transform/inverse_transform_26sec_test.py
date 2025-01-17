# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from pybear.preprocessing.InterceptManager._inverse_transform. \
    _inverse_transform import _inverse_transform

from pybear.preprocessing.InterceptManager.InterceptManager import \
    InterceptManager


from pybear.utilities import nan_mask

import uuid
import numpy as np
import pandas as pd
import scipy.sparse as ss
import dask.array as da
import dask.dataframe as ddf

import pytest







class TestInverseTransform:

    # build an X with duplicates, use IM to manage the constant columns
    # under different parameters (IM transform() should be independently
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
    def _columns(_master_columns, _shape):
        return _master_columns.copy()[:_shape[1]]


    @staticmethod
    @pytest.fixture(scope='function')
    def _const_X(_X_factory, _shape):

        def foo(_has_nan, _format, _dtype, _constants):

            return _X_factory(
                _dupl=None,
                _has_nan=_has_nan,
                _format=_format,
                _dtype=_dtype,
                _constants=_constants,
                _shape=_shape
            )

        return foo


    @staticmethod
    @pytest.fixture(scope='module')
    def _kwargs(_X_factory, _shape):

        return {
            'keep': 'last',
            'equal_nan': True,
            'rtol': 1e-5,
            'atol': 1e-8,
            'n_jobs': -1
        }


    # END fixtures ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    @pytest.mark.parametrize('junk_X', ([], [[]], None, 'junk_string', 3, np.pi))
    def test_rejects_junk_X(self, _const_X, _shape, junk_X, _kwargs):
        # VALIDATION OF X GOING INTO inverse_transform IS HANDLED BY
        # pybear validate_data / _val_X

        FIT_TRFM_X = _const_X(
            _has_nan=False,
            _format='np',
            _dtype='flt',
            _constants={0: 1, 1: 5, _shape[1]-1: np.e}
        )

        TestClass = InterceptManager(**_kwargs)
        _ = TestClass.fit_transform(FIT_TRFM_X)

        with pytest.raises(ValueError):
            TestClass.inverse_transform(junk_X)


    @pytest.mark.parametrize('X_format', ('da', 'ddf'))
    def test_rejects_bad_X(self, _const_X, _shape, X_format, _kwargs):
        # VALIDATION OF X GOING INTO inverse_transform IS HANDLED BY
        # pybear validate_data / _val_X

        FIT_TRFM_X = _const_X(
            _has_nan=False,
            _format='np',
            _dtype='flt',
            _constants={0: 1, 1: 5, _shape[1]-1: np.e}
        )

        if X_format == 'da':
            BAD_FIT_TRFM_X = da.from_array(FIT_TRFM_X)
        elif X_format == 'ddf':
            BAD_FIT_TRFM_X = ddf.from_array(FIT_TRFM_X)
        else:
            raise Exception


        TestClass = InterceptManager(**_kwargs)
        _ = TestClass.fit_transform(FIT_TRFM_X)

        with pytest.raises(TypeError):
            TestClass.inverse_transform(BAD_FIT_TRFM_X)


    @pytest.mark.parametrize('obj_type', ('np', 'pd'))
    @pytest.mark.parametrize('diff_cols', ('more', 'less', 'same'))
    def test_rejects_bad_shape(
        self, _const_X, _kwargs, _shape, obj_type, diff_cols
    ):

        # SHOULD RAISE ValueError WHEN COLUMNS DO NOT EQUAL NUMBER OF
        # RETAINED COLUMNS

        # Build X & TRFM X --- --- --- --- --- --- --- --- --- --- --- ---
        _wip_X = _const_X(
            _has_nan=False,
            _format='np',
            _dtype='flt',
            _constants={0: 1, 1: 5, _shape[1]-1: np.e}
        )

        columns = np.array([str(uuid.uuid4()) for _ in range(_shape[1])])

        TestClass = InterceptManager(**_kwargs)
        TRFM_X = TestClass.fit_transform(_wip_X)
        TRFM_MASK = TestClass.column_mask_
        # END Build X & TRFM X --- --- --- --- --- --- --- --- --- --- ---


        if diff_cols == 'same':
            if obj_type == 'pd':
                TRFM_X = pd.DataFrame(
                    data=TRFM_X, columns=columns[TRFM_MASK]
                )
        elif diff_cols == 'less':
            TRFM_X = TRFM_X[:, :2].copy()
            if obj_type == 'pd':
                TRFM_X = pd.DataFrame(
                    data=TRFM_X, columns=columns[TRFM_MASK][:2]
                )
        elif diff_cols == 'more':
            TRFM_X = np.hstack((TRFM_X.copy(), TRFM_X.copy()))
            if obj_type == 'pd':
                _COLUMNS = np.hstack((
                    columns[TRFM_MASK],
                    np.char.upper(columns[TRFM_MASK])
                ))
                TRFM_X = pd.DataFrame(data=TRFM_X, columns=_COLUMNS)


        # Test the inverse_transform operation ** ** ** ** ** ** ** ** ** **
        if diff_cols == 'same':
            INV_TRFM_X = TestClass.inverse_transform(TRFM_X)
        else:
            with pytest.raises(ValueError):
                TestClass.inverse_transform(TRFM_X)
            pytest.skip(f'cant do more test if inverse_transform call excepted')
        # Test the inverse_transform operation ** ** ** ** ** ** ** ** ** **


        if isinstance(TRFM_X, np.ndarray):
            assert isinstance(INV_TRFM_X, np.ndarray), \
                f"output of inverse_transform() is not a numpy array"
            assert INV_TRFM_X.flags['C_CONTIGUOUS'] is True
        elif isinstance(TRFM_X, pd.core.frame.DataFrame):
            assert isinstance(INV_TRFM_X, pd.core.frame.DataFrame), \
                f"output of inverse_transform() is not a pd dataframe"
        else:
            raise Exception


        assert INV_TRFM_X.shape[0] == TRFM_X.shape[0], \
            f"rows in output of inverse_transform() do not match input rows"
        assert INV_TRFM_X.shape[1] == TestClass.n_features_in_, \
            (f"columns in output of inverse_transform() do not match "
             f"originally fitted columns")

        # convert pd INV_TRFM_X to np for array_equal
        if isinstance(INV_TRFM_X, pd.core.frame.DataFrame):
            INV_TRFM_X = INV_TRFM_X.to_numpy()

        assert np.array_equiv(INV_TRFM_X, _wip_X), \
            f"inverse transform of transformed data does not equal original data"

        assert np.array_equiv(
            TRFM_X.astype(str),
            INV_TRFM_X[:, TestClass.column_mask_].astype(str)
        ), (f"output of inverse_transform() does not reduce back to the output "
            f"of transform()")

        del TRFM_X, TRFM_MASK, INV_TRFM_X, TestClass


    @pytest.mark.parametrize('_dtype', ('flt', 'int', 'str', 'obj', 'hybrid'))
    @pytest.mark.parametrize('_keep',
        (
            'first', 'last', 'random', 'none', 0, 'good_string', lambda x: 0,
            {'Intercept': 1}
        )
    )
    @pytest.mark.parametrize('_has_nan', (True, False))
    @pytest.mark.parametrize('_equal_nan', (True, False))
    @pytest.mark.parametrize('_constants', ('constants1', 'constants2'))
    @pytest.mark.parametrize('_format',
        # run only a few ss representatives to save time
        # 'csr_matrix', 'csc_matrix', 'coo_matrix', 'dia_matrix', 'lil_matrix',
        # 'dok_matrix', 'bsr_matrix', 'csr_array', 'csc_array', 'coo_array',
        # 'dia_array'
        (
            'np', 'pd_with_header', 'pd_without_header', 'csr_matrix',
            'bsr_array'
        )
    )
    def test_accuracy(
        self, _const_X, _dtype, _keep, _has_nan, _equal_nan, _constants,
        _format, _columns, _rtol_atol, _shape
    ):

        # Methodology: transform data, then transform back using
        # inverse_transform. the inverse transform must be equal to the
        # originally fitted data, except for nans. inverse transform
        # cannot infer the presence of nans in the original data.

        if _dtype not in ('flt', 'int') and _format not in ('np', 'pd'):
            pytest.skip(reason=f"scipy sparse cannot take strings")

        if 'pd' not in _format and isinstance(_keep, str) and \
                _keep not in ('first', 'last', 'random', 'none'):
            pytest.skip(reason=f"cannot pass keep as str if X is not pd df")


        if _constants == 'constants1':
            if _dtype in ('int', 'flt'):
                _constants = {0:1, _shape[1]-1: 1}
            else:
                _constants = {0: 'a', _shape[1]-1:'b'}
        elif _constants == 'constants2':
            if _dtype in ('int', 'flt'):
                _constants = {0:1, _shape[1]-1: np.nan}
            else:
                _constants = {0: '1', _shape[1]-1:'nan'}
        else:
            raise Exception


        _base_X = _const_X(
            _has_nan=_has_nan,
            _format='np',
            _dtype=_dtype,
            _constants=_constants
        )


        if _keep == 'good_string':
            if _format == 'pd_with_header':
                _keep = _columns[0]
            elif _format == 'pd_without_header':
                _keep = pd.RangeIndex(start=0, stop=_shape[1], step=1)[0]

        if _format == 'np':
            _X_wip = _base_X
        elif _format == 'pd_with_header':
            _X_wip = pd.DataFrame(
                data=_base_X,
                columns=_columns
            )
        elif _format == 'pd_without_header':
            _X_wip = pd.DataFrame(
                data=_base_X,
                columns=None
            )
        elif _format == 'csr_matrix':
            _X_wip = ss._csr.csr_matrix(_base_X)
        elif _format == 'csc_matrix':
            _X_wip = ss._csc.csc_matrix(_base_X)
        elif _format == 'coo_matrix':
            _X_wip = ss._coo.coo_matrix(_base_X)
        elif _format == 'dia_matrix':
            _X_wip = ss._dia.dia_matrix(_base_X)
        elif _format == 'lil_matrix':
            _X_wip = ss._lil.lil_matrix(_base_X)
        elif _format == 'dok_matrix':
            _X_wip = ss._dok.dok_matrix(_base_X)
        elif _format == 'bsr_matrix':
            _X_wip = ss._bsr.bsr_matrix(_base_X)
        elif _format == 'csr_array':
            _X_wip = ss._csr.csr_array(_base_X)
        elif _format == 'csc_array':
            _X_wip = ss._csc.csc_array(_base_X)
        elif _format == 'coo_array':
            _X_wip = ss._coo.coo_array(_base_X)
        elif _format == 'dia_array':
            _X_wip = ss._dia.dia_array(_base_X)
        elif _format == 'lil_array':
            _X_wip = ss._lil.lil_array(_base_X)
        elif _format == 'dok_array':
            _X_wip = ss._dok.dok_array(_base_X)
        elif _format == 'bsr_array':
            _X_wip = ss._bsr.bsr_array(_base_X)
        else:
            raise Exception

        _IM = InterceptManager(
            keep=_keep,
            rtol=1e-5,
            atol=1e-8,
            equal_nan=_equal_nan,
            n_jobs=1   # leave this at 1 because of confliction
        )

        # fit v v v v v v v v v v v v v v v v v v v v
        except_for_non_const_keep = 0
        if callable(_keep):
            except_for_non_const_keep += 1
        elif isinstance(_keep, int):
            except_for_non_const_keep += 1
        elif isinstance(_keep, str) and \
                _keep not in ('first', 'last', 'random', 'none'):
            except_for_non_const_keep += 1

        if _has_nan and not _equal_nan and except_for_non_const_keep:
            with pytest.raises(ValueError):
                _IM.fit(_X_wip)
            pytest.skip(f"cant do anymore tests without fit")
        else:
            _IM.fit(_X_wip)

        del except_for_non_const_keep
        # fit ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^

        # transform v v v v v v v v v v v v v v v v v v
        _trfm_x = _IM.transform(_X_wip, copy=True)
        # transform ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^

        # inverse transform v v v v v v v v v v v v v v v

        # if keep is a dict, a column was appended during transform.
        # that column needs to be removed before going into _inverse_transform.
        if isinstance(_keep, dict):
            if isinstance(_trfm_x, np.ndarray):
                _trfm_x = _trfm_x[:, :-1]
            elif isinstance(_trfm_x, pd.core.frame.DataFrame):
                _trfm_x = _trfm_x.iloc[:, :-1]
            elif hasattr(_trfm_x, 'toarray'):
                _og_dtype = type(_trfm_x)
                _trfm_x = _trfm_x.tocsc()[:, :-1]
                _trfm_x = _og_dtype(_trfm_x)
                del _og_dtype

        out = _inverse_transform(
            X=_trfm_x,
            _removed_columns=_IM.removed_columns_,
            _feature_names_in=_columns if _format == 'pd_with_header' else None
        )
        # inverse transform ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^


        assert type(out) is type(_X_wip)

        assert out.shape == _X_wip.shape

        if _format == 'pd_with_header':
            assert np.array_equal(out.columns, _columns)
        elif _format == 'pd_without_header':
            assert out.columns.equals(
                pd.RangeIndex(start=0, stop=_shape[1], step=1)
            )


        # iterate over the input X and output X simultaneously, check
        # equality column by column. remember that inverse_transform
        # cannot replicate any nan-likes that may have been in the
        # removed columns in the original data.

        for _og_idx in range(_shape[1]):

            if isinstance(_X_wip, np.ndarray):
                _og_col = _X_wip[:, _og_idx]
                _out_col = out[:, _og_idx]
            elif isinstance(_X_wip, pd.core.frame.DataFrame):
                _og_col = _X_wip.iloc[:, _og_idx].to_numpy()
                _out_col = out.iloc[:, _og_idx].to_numpy()
            elif hasattr(_X_wip, 'toarray'):
                _og_col = _X_wip.tocsc()[:, [_og_idx]].toarray()
                _out_col = out.tocsc()[:, [_og_idx]].toarray()
            else:
                raise Exception


            try:
                _og_col[np.logical_not(nan_mask(_og_col))].astype(np.float64)
                is_num = True
            except:
                is_num = False


            if is_num:

                # allclose is not calling equal on two identical vectors,
                # one w nans and the other without, even with equal_nan.
                # also verified this behavior externally.
                # _og_col may or may not have the nans, but _out_col cannot.
                # put nans into _out_col to get around this.

                MASK = nan_mask(_og_col)
                if np.any(MASK):
                    _og_col[MASK] = np.nan
                    _out_col[MASK] = np.nan

                assert np.allclose(
                    _out_col.astype(np.float64),
                    _og_col.astype(np.float64),
                    equal_nan=True
                )
            else:

                # need to account for nans that may be in _og_col but
                # cant be in _out_col
                MASK = nan_mask(_og_col)
                if np.any(MASK):
                    _og_col[MASK] = 'nan'
                    _out_col[MASK] = 'nan'

                assert np.array_equal(_out_col, _og_col)
























