# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

import numpy as np
import scipy.sparse as ss



@pytest.mark.parametrize('_format',
    ('csr_matrix', 'csc_matrix', 'coo_matrix', 'dia_matrix', 'lil_matrix',
     'dok_matrix', 'bsr_matrix', 'csr_array', 'csc_array', 'coo_array',
     'dia_array', 'lil_array', 'dok_array', 'bsr_array')
)
def test_scipy_sparse_slicing(_format):

    _shape = (5,3)

    _X = np.random.randint(0, 10, _shape)

    if _format == 'csr_matrix':
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


    if _format in ('coo_matrix', 'dia_matrix', 'bsr_matrix', 'coo_array',
                   'dia_array', 'bsr_array'):
        with pytest.raises((TypeError, NotImplementedError)):
            _X_wip[:, [0, _shape[1] - 1]]
    else:
        _X_wip[:, [0, _shape[1]-1]]





