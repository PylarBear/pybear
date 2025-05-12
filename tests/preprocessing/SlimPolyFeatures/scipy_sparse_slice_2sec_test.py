# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

import scipy.sparse as ss



@pytest.mark.parametrize('_format',
    ('csr_matrix', 'csc_matrix', 'coo_matrix', 'dia_matrix', 'lil_matrix',
     'dok_matrix', 'bsr_matrix', 'csr_array', 'csc_array', 'coo_array',
     'dia_array', 'lil_array', 'dok_array', 'bsr_array')
)
def test_scipy_sparse_slicing(_format, X_np, _shape):


    if _format == 'csr_matrix':
        _X_wip = ss._csr.csr_matrix(X_np)
    elif _format == 'csc_matrix':
        _X_wip = ss._csc.csc_matrix(X_np)
    elif _format == 'coo_matrix':
        _X_wip = ss._coo.coo_matrix(X_np)
    elif _format == 'dia_matrix':
        _X_wip = ss._dia.dia_matrix(X_np)
    elif _format == 'lil_matrix':
        _X_wip = ss._lil.lil_matrix(X_np)
    elif _format == 'dok_matrix':
        _X_wip = ss._dok.dok_matrix(X_np)
    elif _format == 'bsr_matrix':
        _X_wip = ss._bsr.bsr_matrix(X_np)
    elif _format == 'csr_array':
        _X_wip = ss._csr.csr_array(X_np)
    elif _format == 'csc_array':
        _X_wip = ss._csc.csc_array(X_np)
    elif _format == 'coo_array':
        _X_wip = ss._coo.coo_array(X_np)
    elif _format == 'dia_array':
        _X_wip = ss._dia.dia_array(X_np)
    elif _format == 'lil_array':
        _X_wip = ss._lil.lil_array(X_np)
    elif _format == 'dok_array':
        _X_wip = ss._dok.dok_array(X_np)
    elif _format == 'bsr_array':
        _X_wip = ss._bsr.bsr_array(X_np)
    else:
        raise Exception


    if _format in ('coo_matrix', 'dia_matrix', 'bsr_matrix', 'coo_array',
                   'dia_array', 'bsr_array'):
        with pytest.raises((TypeError, NotImplementedError)):
            _X_wip[:, [0, _shape[1] - 1]]
    else:
        _X_wip[:, [0, _shape[1]-1]]





