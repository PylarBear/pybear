# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import numpy as np
import pandas as pd
import scipy.sparse as ss
import dask.array as da
import dask.dataframe as ddf
import dask_expr._collection as ddf2

from pybear.preprocessing.MinCountTransformer._set_output._get_og_obj_type import \
    _get_og_obj_type

import pytest




class TestGetOGObjType:

    def test_OBJECT_rejects_junk(self):
        pass
        # pizza finish


    def test_original_obj_type_rejects_junk(self):
        pass
        # pizza finish


    def test_accuracy(self):

        X = np.random.randint(0, 10, (10, 10))

        assert _get_og_obj_type(None, None) is None
        assert _get_og_obj_type(np.array(X), None) == 'numpy_array'
        assert _get_og_obj_type(pd.DataFrame(X), None) == 'pandas_dataframe'
        assert _get_og_obj_type(pd.Series(X[:, 0]), None) == 'pandas_series'
        assert _get_og_obj_type(da.from_array(X), None) == 'dask_array'
        assert _get_og_obj_type(ddf.from_array(X), None) == 'dask_dataframe'
        assert _get_og_obj_type(ddf.from_array(X[:, 0].reshape((-1, 1))).squeeze(), None) == 'dask_series'
        assert _get_og_obj_type(ddf2.DataFrame(X), None) == 'dask_dataframe'
        assert _get_og_obj_type(ddf2.Series(X[:, 0]), None) == 'dask_series'
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


