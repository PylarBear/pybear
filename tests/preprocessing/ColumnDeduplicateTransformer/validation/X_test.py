# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

from pybear.preprocessing._ColumnDeduplicateTransformer._validation._X \
    import _val_X



def test_X_cannot_be_none():

    with pytest.raises(TypeError):
        _val_X(None)


@pytest.mark.parametrize('X_format',
     ('np', 'pd', 'pl', 'csr_array', 'csc_array', 'dok_array', 'bsr_matrix', 'bsr_array')
)
def test_accepts_np_pd_ss(_X_factory, _shape, X_format):

    # accepts all scipy sparse

    _X = _X_factory(
        _format=X_format,
        _dtype='flt',
        _shape=_shape
    )

    _val_X(_X)






