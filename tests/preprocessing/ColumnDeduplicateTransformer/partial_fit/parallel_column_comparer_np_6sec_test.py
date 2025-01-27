# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


# IMPORTANT!
# this is not sparse so this uses _parallel_column_comparer!

from pybear.preprocessing.ColumnDeduplicateTransformer._partial_fit. \
    _parallel_column_comparer import _parallel_column_comparer


import pytest




class TestNpColumnComparer:


    @staticmethod
    @pytest.fixture(scope='module')
    def _shape():
        return (1000, 2)


    # np cant be int if using nans
    @pytest.mark.parametrize('_dtype1', ('flt', 'str', 'obj'))
    @pytest.mark.parametrize('_dtype2', ('flt', 'str', 'obj'))
    @pytest.mark.parametrize('_has_nan', (True, False))
    @pytest.mark.parametrize('_equal_nan', (True, False))
    def test_accuracy(
        self, _X_factory, _dtype1, _dtype2, _has_nan, _equal_nan, _shape
    ):

        # a sneaky trick here. _X_factory peppers nans after propagating
        # duplicates. which means nans are likely to be different on every
        # column. so if create a 2 column array and both columns are the
        # same, then both will be identical except for the nans.


        _X_flt = _X_factory(
            _dupl=[[0,1]],
            _format='np',
            _dtype='flt',
            _has_nan=_has_nan,
            _columns=None,
            _zeros=0.33,
            _shape=_shape
        )

        _X_str = _X_factory(
            _dupl=[[0,1]],
            _format='np',
            _dtype='str',
            _has_nan=_has_nan,
            _columns=None,
            _zeros=0.33,
            _shape=_shape
        )

        _X_obj = _X_factory(
            _dupl=[[0,1]],
            _format='np',
            _dtype='obj',
            _has_nan=_has_nan,
            _columns=None,
            _zeros=0.33,
            _shape=_shape
        )

        if _dtype1 == 'flt':
            _X1 = _X_flt[:,0].ravel()
        elif _dtype1 == 'str':
            _X1 = _X_str[:,0].ravel()
        elif _dtype1 == 'obj':
            _X1 = _X_obj[:,0].ravel()
        else:
            raise Exception

        if _dtype2 == 'flt':
            _X2 = _X_flt[:,1].ravel()
        elif _dtype2 == 'str':
            _X2 = _X_str[:,1].ravel()
        elif _dtype2 == 'obj':
            _X2 = _X_obj[:,1].ravel()
        else:
            raise Exception


        _are_equal = _parallel_column_comparer(
            _X1, _X2, _rtol=1e-5, _atol=1e-8, _equal_nan=_equal_nan
        )

        if _dtype1 == _dtype2:
            if _equal_nan or not _has_nan:
                assert _are_equal
            elif not _equal_nan:
                assert not _are_equal
        else:
            assert not _are_equal
























