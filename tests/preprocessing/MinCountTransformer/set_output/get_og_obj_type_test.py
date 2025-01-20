# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import numpy as np
import pandas as pd
import dask.array as da
import dask.dataframe as ddf
import scipy.sparse as ss

from pybear.preprocessing.MinCountTransformer._set_output._get_og_obj_type import \
    _get_og_obj_type

import pytest




class TestGetOGObjType:


    # def _get_og_obj_type(
    #     OBJECT: Union[XContainer, YContainer],
    #     _original_obj_type: Union[str, None]  # use self._x_original_obj_type
    # ) -> str:


    @staticmethod
    @pytest.fixture(scope='module')
    def _X_np():

        return np.random.uniform(0, 1, (10, 10))


    @staticmethod
    @pytest.fixture(scope='module')
    def _good_og_dtype():
        return 'numpy_array'


    @pytest.mark.parametrize('junk_OBJECT',
        (-2.7, -1, 0, 1, 2.7, True, 'junk', [1, 2], {'a': 1}, lambda x: x)
    )
    def test_OBJECT_rejects_junk(self, junk_OBJECT, _good_og_dtype):

        with pytest.raises(TypeError):
            _get_og_obj_type(
                junk_OBJECT,
                _good_og_dtype
            )


    @pytest.mark.parametrize('bad_OBJECT', ('da_array', 'dask_ddf', 'dask_series'))
    def test_OBJECT_rejects_bad(self, bad_OBJECT, _X_np, _good_og_dtype):

        if bad_OBJECT == 'da_array':
            _X_wip = da.from_array(_X_np)
        elif bad_OBJECT == 'dask_ddf':
            _X_wip = ddf.from_array(_X_np)
        elif bad_OBJECT == 'dask_series':
            _X_wip = ddf.from_array(_X_np[:, 0]).squeeze()
        else:
            raise Exception

        with pytest.raises(TypeError):
            _get_og_obj_type(
                _X_wip,
                _good_og_dtype
            )


    @pytest.mark.parametrize('OBJECT', ('np_array', 'pd_df', 'pd_series', None))
    def test_OBJECT_accepts_good(self, OBJECT, _X_np, _good_og_dtype):

        if OBJECT == 'np_array':
            _X_wip = _X_np
        elif OBJECT == 'pd_df':
            _X_wip = pd.DataFrame(_X_np)
        elif OBJECT == 'pd_series':
            _X_wip = pd.Series(_X_np[:, 0])
        elif OBJECT is None:
            _X_wip = None
        else:
            raise Exception

        if OBJECT is None:
            assert isinstance(_get_og_obj_type(_X_wip, _good_og_dtype), type(None))
        else:
            assert isinstance(_get_og_obj_type(_X_wip, _good_og_dtype), str)


    @pytest.mark.parametrize('junk_type',
        (-2.7, -1, 0, 1, 2.7, True, [1, 2], {'a': 1}, lambda x: x)
    )
    def test_original_obj_type_rejects_junk(self, _X_np, junk_type):

        with pytest.raises(TypeError):
            _get_og_obj_type(
                _X_np,
                junk_type
            )


    @pytest.mark.parametrize('bad_type', ('garbage', 'junk', 'trash'))
    def test_original_obj_type_rejects_bad(self, _X_np, bad_type):

        with pytest.raises(ValueError):
            _get_og_obj_type(
                _X_np,
                bad_type
            )


    def test_original_obj_type_accepts_good(self, _X_np, _good_og_dtype):

        assert _get_og_obj_type(_X_np, _good_og_dtype) == _good_og_dtype
        assert _get_og_obj_type(_X_np, None) == _good_og_dtype


    def test_accuracy(self):

        X = np.random.randint(0, 10, (10, 10))

        assert _get_og_obj_type(None, None) is None
        assert _get_og_obj_type(np.array(X), None) == 'numpy_array'
        assert _get_og_obj_type(pd.DataFrame(X), None) == 'pandas_dataframe'
        assert _get_og_obj_type(pd.Series(X[:, 0]), None) == 'pandas_series'
        assert _get_og_obj_type(ss.csr_matrix(X), None) == 'scipy_sparse_csr_matrix'
        assert _get_og_obj_type(ss.csr_array(X), None) == 'scipy_sparse_csr_array'
        assert _get_og_obj_type(ss.csc_matrix(X), None) == 'scipy_sparse_csc_matrix'
        assert _get_og_obj_type(ss.csc_array(X), None) == 'scipy_sparse_csc_array'
        assert _get_og_obj_type(ss.coo_matrix(X), None) == 'scipy_sparse_coo_matrix'
        assert _get_og_obj_type(ss.coo_array(X), None) == 'scipy_sparse_coo_array'
        assert _get_og_obj_type(ss.dia_matrix(X), None) == 'scipy_sparse_dia_matrix'
        assert _get_og_obj_type(ss.dia_array(X), None) == 'scipy_sparse_dia_array'
        assert _get_og_obj_type(ss.lil_matrix(X), None) == 'scipy_sparse_lil_matrix'
        assert _get_og_obj_type(ss.lil_array(X), None) == 'scipy_sparse_lil_array'
        assert _get_og_obj_type(ss.dok_matrix(X), None) == 'scipy_sparse_dok_matrix'
        assert _get_og_obj_type(ss.dok_array(X), None) == 'scipy_sparse_dok_array'
        assert _get_og_obj_type(ss.bsr_matrix(X), None) == 'scipy_sparse_bsr_matrix'
        assert _get_og_obj_type(ss.bsr_array(X), None) == 'scipy_sparse_bsr_array'








