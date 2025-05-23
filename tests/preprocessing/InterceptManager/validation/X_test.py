# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

from pybear.preprocessing._InterceptManager._validation._X import _val_X



def test_X_cannot_be_none():

    with pytest.raises(TypeError):
        _val_X(None)


@pytest.mark.parametrize('X_format', (list, tuple))
def test_rejects_py_builtin(_X_factory, X_format, _shape):

    _X = _X_factory(_dupl=None, _format='np', _shape=_shape)

    _X = X_format(map(X_format, _X))

    with pytest.raises(TypeError):
        _val_X(_X)


@pytest.mark.parametrize('X_format',
    ('np', 'pd', 'pl', 'csr_array', 'csc_array', 'bsr_array')
)
def test_accepts_np_pd_ss(_X_factory, X_format, _shape):

    _X = _X_factory(_dupl=None, _format=X_format, _shape=_shape)

    _val_X(_X)






