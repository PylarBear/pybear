# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


# IMPORTANT!
# this is not sparse so this uses _parallel_column_comparer!


import pytest

from pybear.preprocessing._ColumnDeduplicateTransformer._partial_fit. \
    _parallel_column_comparer import _parallel_column_comparer



class TestPdColumnComparer:


    @pytest.mark.parametrize('_dtype1', ('flt', 'int', 'str', 'obj'))
    @pytest.mark.parametrize('_dtype2', ('flt', 'int', 'str', 'obj'))
    @pytest.mark.parametrize('_has_nan', (True, False))
    @pytest.mark.parametrize('_equal_nan', (True, False))
    def test_accuracy(
        self, _X_factory, _dtype1, _dtype2, _has_nan, _equal_nan
    ):

        # a sneaky trick here. _X_factory peppers nans after propagating
        # duplicates. which means nans are likely to be different on every
        # column. so if create a 2 column array and both columns are the
        # same, then both will be identical except for the nans.

        _shape = (1000, 2)

        _X_flt = _X_factory(
            _dupl=[[0,1]],
            _format='pd',   # pizza pl?
            _dtype='flt',
            _has_nan=_has_nan,
            _columns=['a','b'],
            _zeros=0.33,
            _shape=_shape
        )

        _X_int = _X_factory(
            _dupl=[[0,1]],
            _format='pd',   # pizza pl?
            _dtype='int',
            _has_nan=_has_nan,
            _columns=['c','d'],
            _zeros=0.33,
            _shape=_shape
        )

        _X_str = _X_factory(
            _dupl=[[0,1]],
            _format='pd',   # pizza pl?
            _dtype='str',
            _has_nan=_has_nan,
            _columns=['e','f'],
            _zeros=0.33,
            _shape=_shape
        )

        _X_obj = _X_factory(
            _dupl=[[0,1]],
            _format='pd',    # pizza pl?
            _dtype='obj',
            _has_nan=_has_nan,
            _columns=['g','h'],
            _zeros=0.33,
            _shape=_shape
        )

        if _dtype1 == 'flt':
            _X1 = _X_flt.iloc[:, [0]]
        elif _dtype1 == 'int':
            _X1 = _X_int.iloc[:, [0]]
        elif _dtype1 == 'str':
            _X1 = _X_str.iloc[:, [0]]
        elif _dtype1 == 'obj':
            _X1 = _X_obj.iloc[:, [0]]
        else:
            raise Exception

        if _dtype2 == 'flt':
            _X2 = _X_flt.iloc[:, [1]]
        elif _dtype2 == 'int':
            _X2 = _X_int.iloc[:, [1]]
        elif _dtype2 == 'str':
            _X2 = _X_str.iloc[:, [1]]
        elif _dtype2 == 'obj':
            _X2 = _X_obj.iloc[:, [1]]
        else:
            raise Exception


        _are_equal = _parallel_column_comparer(
            _X1.to_numpy(), _X2.to_numpy(), _rtol=1e-5, _atol=1e-8, _equal_nan=_equal_nan
        )[0]

        if _dtype1 == _dtype2:
            if _equal_nan or not _has_nan:
                assert _are_equal
            elif not _equal_nan:
                assert not _are_equal
        else:
            assert not _are_equal






