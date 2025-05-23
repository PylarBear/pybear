# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest



@pytest.mark.parametrize('_format',
    ('csr_matrix', 'csc_matrix', 'coo_matrix', 'dia_matrix', 'lil_matrix',
     'dok_matrix', 'bsr_matrix', 'csr_array', 'csc_array', 'coo_array',
     'dia_array', 'lil_array', 'dok_array', 'bsr_array')
)
def test_scipy_sparse_slicing(_X_factory, _format, _shape):

    _X_wip = _X_factory(
        _dupl=None,
        _has_nan=False,
        _format=_format,
        _dtype='flt',
        _columns=None,
        _constants=None,
        _noise=0,
        _zeros=None,
        _shape=_shape
    )


    if _format in ('coo_matrix', 'dia_matrix', 'bsr_matrix', 'coo_array',
                   'dia_array', 'bsr_array'):
        with pytest.raises((TypeError, NotImplementedError)):
            _X_wip[:, [0, _shape[1] - 1]]
    else:
        _X_wip[:, [0, _shape[1]-1]]





