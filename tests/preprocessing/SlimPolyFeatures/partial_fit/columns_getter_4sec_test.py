# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

import numpy as np
import scipy.sparse as ss

from pybear.preprocessing._SlimPolyFeatures._partial_fit._columns_getter \
    import _columns_getter


# only need to test numeric data, SPF blocks all non-numeric data

# this mark needs to stay here because _X_num fixture needs it
@pytest.mark.parametrize('_has_nan', (True, False), scope='module')
class TestColumnsGetter:

    # fixtures ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

    @staticmethod
    @pytest.fixture(scope='module')
    def _X_num(_X_factory, _shape, _has_nan):

        return _X_factory(
            _dupl=None,
            _format='np',
            _dtype='flt',
            _has_nan=_has_nan,
            _columns=None,
            _shape=_shape
        )

    # END fixtures ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **


    @pytest.mark.parametrize('_col_idxs',
        (0, 20, (0,), (20,), (0,1), (0,20), (100,200))
    )
    def test_rejects_idx_out_of_col_range(
        self, _X_num, _has_nan, _col_idxs, _shape
    ):

        _out_of_range = False
        try:
            tuple(_col_idxs)
            # is a tuple
            for _idx in _col_idxs:
                if _idx not in range(_shape[1]):
                    _out_of_range = True
        except:
            # is int
            if _col_idxs not in range(_shape[1]):
                _out_of_range = True


        if _out_of_range:
            with pytest.raises(AssertionError):
                _columns_getter(_X_num, _col_idxs)
        else:
            _columns = _columns_getter(_X_num, _col_idxs)


    @pytest.mark.parametrize('_format',
        (
        'np', 'pd', 'pl', 'csr_matrix', 'csc_matrix', 'coo_matrix',
        'dia_matrix', 'lil_matrix', 'dok_matrix', 'bsr_matrix', 'csr_array',
        'csc_array', 'coo_array', 'dia_array', 'lil_array', 'dok_array',
        'bsr_array'
        )
    )
    @pytest.mark.parametrize('_col_idxs',
        (0, 1, 2, (0,), (1,), (2,), (0,1), (0,2), (1,2), (0,1,2))
    )
    def test_accuracy(
        self, _X_factory, _shape, _format, _col_idxs, _X_num, _columns
    ):

        # _columns_getter only allows ss that are indexable
        # coo, dia, & bsr matrix/array are blocked. should raise.

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

        if _format == 'np':
            _base_X = _X_wip
        elif _format in ['pd', 'pl']:
            _base_X = _X_wip.to_numpy()
        elif hasattr(_X_wip, 'toarray'):
            _base_X = _X_wip.toarray()

        # ensure _col_idxs, when tuple, is sorted, _columns_getter requires this,
        # make sure any careless changes made to this test are handled.
        try:
            iter(_col_idxs)
            _col_idxs = tuple(sorted(list(_col_idxs)))
        except:
            pass

        # pass _col_idxs as given (int or tuple) to _columns getter
        # verify _columns_getter rejects coo, dia, and bsr
        if isinstance(_X_wip,
            (ss.coo_matrix, ss.coo_array,
             ss.dia_matrix, ss.dia_array,
             ss.bsr_matrix, ss.bsr_array)
        ):
            with pytest.raises(AssertionError):
                _columns_getter(_X_wip, _col_idxs)
            pytest.skip(reason=f"cant do anymore tests after exception")
        else:
            _columns = _columns_getter(_X_wip, _col_idxs)

        assert isinstance(_columns, np.ndarray)

        # now that _columns getter has seen the given _col_idxs, covert all
        # given _col_idxs to tuple to make _X[:, _col_idxs] slice right, below
        try:
            len(_col_idxs)  # except if is integer
        except: # if is integer change to tuple
            _col_idxs = (_col_idxs,)

        assert _columns.shape[1] == len(_col_idxs)

        # since all the various _X_wips came from _X_num, just use _X_num
        # to referee whether _columns_getter pulled the correct columns
        # from _X_wip
        assert np.array_equal(_columns, _base_X[:, _col_idxs], equal_nan=True)




