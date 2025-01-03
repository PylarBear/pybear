# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from pybear.base._cast_to_ndarray import cast_to_ndarray
from pybear.utilities._nan_masking import nan_mask

import numpy as np
import pandas as pd
import scipy.sparse as ss
import dask.array as da
import dask.dataframe as ddf
from dask_expr._collection import DataFrame as ddf2


import pytest


pytest.skip(reason='pizza needs to finish', allow_module_level=True)


class TestCastToNDArray:

    # shape must be preserved
    # original dtype must be preserved



    @staticmethod
    @pytest.fixture(scope='module')
    def _shape():
        return (20, 13)


    @staticmethod
    @pytest.fixture(scope='module')
    def _columns(_master_columns, _shape):
        return _master_columns.copy()[:_shape[1]]


    @pytest.mark.parametrize('junk_X',
        (-2.7, -1, 0, 1, 2, True, False, None, 'junk', {'a': 1}, lambda x: x)
    )
    def test_blocks_non_array_like(self, junk_X):

        with pytest.raises(TypeError):
            cast_to_ndarray(junk_X)


    def test_blocks_other_non_descript(self):

        with pytest.raises(TypeError):
            cast_to_ndarray(np.recarray((1,2,3), dtype=np.float64))

        with pytest.raises(TypeError):
            cast_to_ndarray(np.ma.masked_array([1,2,3]))


    @pytest.mark.parametrize('_format',
         (
             'np', 'pd_series', 'pd_df', 'csr_matrix', 'csc_matrix', 'coo_matrix',
             'dia_matrix', 'lil_matrix', 'dok_matrix', 'bsr_matrix', 'csr_array',
             'csc_array', 'coo_array', 'dia_array', 'lil_array', 'dok_array',
             'bsr_array', 'dask_array', 'dask_series', 'dask_ddf'
         )
    )
    @pytest.mark.parametrize('_has_nan', (True, False))
    @pytest.mark.parametrize('_dtype, _sub_dtype',
        (
            ('flt', np.float64),
            ('int', np.uint8),
            ('int', np.uint16),
            ('int', np.uint32),
            ('int', np.uint64),
            ('str', '<U10'),
            ('obj', object)
         )
    )
    def test_accuracy(
        self, _X_factory, _format, _has_nan, _dtype, _sub_dtype, _columns, _shape
    ):

        # skip impossible conditions -- -- -- -- -- -- -- -- -- -- -- --

        if _has_nan and _dtype == 'int':
            pytest.skip(reason=f"int dtypes cant have nan-likes")

        non_ss = [
            'np', 'pd_series', 'pd_df', 'dask_array', 'dask_series', 'dask_ddf'
        ]
        if _format not in non_ss and _dtype not in ['int', 'flt']:
            pytest.skip(reason=f"scipy sparse can only take numeric")

        # if any df and str dtype, that is just forced over to object dtype
        if _format in ['pd_series', 'pd_df', 'dask_series', 'dask_ddf']:
            if _dtype == 'str':
                pytest.skip(reason=f"pandas forces str over to object")

        # END skip impossible conditions -- -- -- -- -- -- -- -- -- -- --


        # for all but pd series & df, construct the base as np then
        # convert to np, ss, da, ddf
        if _format not in ['pd_series', 'pd_df', 'dask_series', 'dask_ddf']:
            _X_base_np = _X_factory(
                _dupl=None,
                _has_nan=_has_nan,
                _format='np',
                _dtype=_dtype,
                _columns=_columns,
                _constants=None,
                _zeros=None,
                _shape=_shape
            )
        # want to be able to capture the effects of junky pd nan-likes.
        # so construct the pd-based things separately to let _X_factory
        # put the junky nan-likes in.
        elif _format in ['pd_series', 'pd_df', 'dask_series', 'dask_ddf']:
            _X_base_pd = _X_factory(
                _dupl=None,
                _has_nan=_has_nan,
                _format='pd',
                _dtype=_dtype,
                _columns=_columns,
                _constants=None,
                _zeros=None,
                _shape=_shape
            )


        if _format == 'np':
            _X_wip = _X_base_np
        elif _format == 'pd_series':
            _X_wip =_X_base_pd
            _X_wip = _X_wip.iloc[:, 0].squeeze()
        elif _format == 'pd_df':
            _X_wip = _X_base_pd
        elif _format == 'csr_matrix':
            _X_wip = ss._csr.csr_matrix(_X_base_np)
        elif _format == 'csc_matrix':
            _X_wip = ss._csc.csc_matrix(_X_base_np)
        elif _format == 'coo_matrix':
            _X_wip = ss._coo.coo_matrix(_X_base_np)
        elif _format == 'dia_matrix':
            _X_wip = ss._dia.dia_matrix(_X_base_np)
        elif _format == 'lil_matrix':
            _X_wip = ss._lil.lil_matrix(_X_base_np)
        elif _format == 'dok_matrix':
            _X_wip = ss._dok.dok_matrix(_X_base_np)
        elif _format == 'bsr_matrix':
            _X_wip = ss._bsr.bsr_matrix(_X_base_np)
        elif _format == 'csr_array':
            _X_wip = ss._csr.csr_array(_X_base_np)
        elif _format == 'csc_array':
            _X_wip = ss._csc.csc_array(_X_base_np)
        elif _format == 'coo_array':
            _X_wip = ss._coo.coo_array(_X_base_np)
        elif _format == 'dia_array':
            _X_wip = ss._dia.dia_array(_X_base_np)
        elif _format == 'lil_array':
            _X_wip = ss._lil.lil_array(_X_base_np)
        elif _format == 'dok_array':
            _X_wip = ss._dok.dok_array(_X_base_np)
        elif _format == 'bsr_array':
            _X_wip = ss._bsr.bsr_array(_X_base_np)
        elif _format == 'dask_array':
            _X_wip = da.from_array(_X_base_np, chunks=_shape)
        elif _format == 'dask_series':
            _X_wip = ddf.from_pandas(_X_base_pd)
            _X_wip = _X_wip.iloc[:, 0].squeeze()
        elif _format == 'dask_ddf':
            _X_wip = ddf.from_pandas(_X_base_pd)
        else:
            raise Exception


        # set the sub-dtype here then check after cast to prove
        # og dtype is preserved
        try:
            # this is excepting when trying to put float64 dtype on pd
            # objects that have str nan-likes in them. what this means is
            # that when _X_factory is building pd with junky nan-likes
            # the dtype is being changed to object. so to do this test
            # when that happens, replace all the junky nan-likes with
            # np.nan then set the dtype to float64.
            _X_wip = _X_wip.astype(_sub_dtype)
        except:
            _X_wip[nan_mask(_X_wip)] = np.nan
            _X_wip = _X_wip.astype(_sub_dtype)

        # v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v
        out = cast_to_ndarray(_X_wip)
        # ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^


        assert isinstance(out, np.ndarray)

        assert out.dtype == _sub_dtype

        if _format in ['pd_series', 'dask_series']:
            assert out.shape == (_shape[0], )
        else:
            assert out.shape == _shape


        # check 'out' against the original base X to prove values were preserved

        if _format in ['pd_series', 'dask_series']:
            ref = _X_base_pd.iloc[:, 0].to_numpy()
        elif _format in ['pd_df', 'dask_ddf']:
            ref = _X_base_pd.to_numpy()
        else:
            ref = _X_base_np

        if _dtype == 'flt':

                assert np.array_equal(out, ref, equal_nan=True)
        else:
            # str NA causing
            # TypeError: boolean value of NA is ambiguous

            not_nan_mask = np.logical_not(nan_mask(ref))

            assert np.array_equal(
                out[not_nan_mask],
                ref[not_nan_mask]
            )















