# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.utilities._scipy_sparse_preslice_handle import \
    scipy_sparse_preslice_handle as ssph

import uuid

import numpy as np
import pandas as pd
import scipy.sparse as ss
import dask.array as da
import dask.dataframe as ddf

import pytest




class TestSSColumnSlice:

    # there is no validation of the incoming object that it is array-like

    @staticmethod
    @pytest.fixture(scope='module')
    def _shape():
        return (50, 37)


    @pytest.mark.parametrize('X_format',
        (
        'np', 'pd', 'csr_matrix', 'csc_matrix', 'coo_matrix', 'dia_matrix',
        'lil_matrix', 'dok_matrix', 'bsr_matrix', 'csr_array', 'csc_array',
        'coo_array', 'dia_array', 'lil_array', 'dok_array', 'bsr_array',
        'dask_array', 'dask_dataframe'
        )
    )
    def test_accuracy(self, X_format, _shape):

        _base_X = np.random.uniform(0, 1, _shape)

        if X_format == 'np':
            _X_wip = _base_X
        elif X_format == 'pd':
            _X_wip = pd.DataFrame(
                data=_base_X,
                columns=[str(uuid.uuid4)[:4] for _ in range(_shape[1])]
            )
        elif X_format == 'csr_matrix':
            _X_wip = ss._csr.csr_matrix(_base_X)
        elif X_format == 'csc_matrix':
            _X_wip = ss._csc.csc_matrix(_base_X)
        elif X_format == 'coo_matrix':
            _X_wip = ss._coo.coo_matrix(_base_X)
        elif X_format == 'dia_matrix':
            _X_wip = ss._dia.dia_matrix(_base_X)
        elif X_format == 'lil_matrix':
            _X_wip = ss._lil.lil_matrix(_base_X)
        elif X_format == 'dok_matrix':
            _X_wip = ss._dok.dok_matrix(_base_X)
        elif X_format == 'bsr_matrix':
            _X_wip = ss._bsr.bsr_matrix(_base_X)
        elif X_format == 'csr_array':
            _X_wip = ss._csr.csr_array(_base_X)
        elif X_format == 'csc_array':
            _X_wip = ss._csc.csc_array(_base_X)
        elif X_format == 'coo_array':
            _X_wip = ss._coo.coo_array(_base_X)
        elif X_format == 'dia_array':
            _X_wip = ss._dia.dia_array(_base_X)
        elif X_format == 'lil_array':
            _X_wip = ss._lil.lil_array(_base_X)
        elif X_format == 'dok_array':
            _X_wip = ss._dok.dok_array(_base_X)
        elif X_format == 'bsr_array':
            _X_wip = ss._bsr.bsr_array(_base_X)
        elif X_format == 'dask_array':
            _X_wip = da.array(_base_X)
        elif X_format == 'dask_dataframe':
            _X_wip = ddf.from_dask_array(
                da.array(_base_X),
                columns=[str(uuid.uuid4)[:4] for _ in range(_shape[1])]
            )
        else:
            raise Exception


        if isinstance(_X_wip,
            (ss.coo_matrix, ss.dia_matrix, ss.bsr_matrix,
             ss.coo_array, ss.dia_array, ss.bsr_array)
        ):

            with pytest.warns() as _warn:
                out = ssph(_X_wip)

            assert isinstance(out, ss.csc_array)

            exp = ("to avoid this, pass your sparse data as csr, csc, "
                f"lil, or dok.")

            any(exp in str(warning.message) for warning in _warn)

        else:

            out = ssph(_X_wip)

            assert isinstance(out, type(_X_wip))









