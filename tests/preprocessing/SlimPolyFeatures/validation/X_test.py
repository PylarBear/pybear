# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#




from pybear.preprocessing._SlimPolyFeatures._validation._X import _val_X

import numpy as np
import pandas as pd
import scipy.sparse as ss
import dask.array as da
import dask.dataframe as ddf


import pytest



class TestValX:


    @staticmethod
    @pytest.fixture(scope='module')
    def np_X():
        return np.random.randint(0, 10, (5, 3))


    # interaction_only ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    @pytest.mark.parametrize('junk_intx_only',
        (-2.7, -1, 0, 1, 2.7, None, 'junk', [0,1], (0,1), {'a':1}, lambda x: x)
    )
    def test_rejects_junk_intx_only(self, np_X, junk_intx_only):

        with pytest.raises(AssertionError):
            _val_X(np_X, _interaction_only=junk_intx_only)


    @pytest.mark.parametrize('_intx_only', (True, False))
    def test_accepts_bool_intx_only(self, np_X, _intx_only):

        assert _val_X(np_X, _interaction_only=_intx_only) is None

    # END interaction_only ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **


    # X format ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
    @pytest.mark.parametrize('junk_X',
        (-2.7, -1, 0, 1, 2.7, True, None, 'junk', [0,1], {'a':1}, lambda x: x)
    )
    @pytest.mark.parametrize('_intx_only', (True, False))
    def test_rejects_junk_X(self, junk_X, _intx_only):

        with pytest.raises(TypeError):
            _val_X(junk_X, _interaction_only=_intx_only)


    @pytest.mark.parametrize('X_format', ('dask_array', 'dask_dataframe'))
    @pytest.mark.parametrize('_intx_only', (True, False))
    def test_rejects_bad_X(self, X_format, _intx_only):

        if X_format == 'dask_array':
            bad_X = da.random.randint(0, 10, (5, 3))
        elif X_format == 'dask_dataframe':
            bad_X = ddf.DataFrame(da.random.randint(0, 10, (5,3)))
        else:
            raise Exception

        with pytest.raises(TypeError):
            _val_X(bad_X, _interaction_only=_intx_only)


    @pytest.mark.parametrize('X_format', ('np', 'pd', 'csc_matrix', 'csc_array'))
    @pytest.mark.parametrize('_intx_only', (True, False))
    def test_accepts_good_X(self, np_X, X_format, _intx_only):

        if X_format == 'np':
            _X = np_X
        elif X_format == 'pd':
            _X = pd.DataFrame(np_X)
        elif X_format == 'csc_matrix':
            _X = ss.csc_matrix(np_X)
        elif X_format == 'csc_array':
            _X = ss.csc_array(np_X)
        else:
            raise Exception


        assert _val_X(_X, _interaction_only=_intx_only) is None

    # END X format ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    @pytest.mark.parametrize('_columns', (1, 2, 3))
    @pytest.mark.parametrize('_intx_only', (True, False))
    def test_quirks_of_shape_and_intx_only(self, _columns, _intx_only):

        # if interaction only must have at least 2 columns
        # if not interaction only can have 1 column

        _X = np.random.randint(0, 10, (10, _columns))

        if _intx_only and _columns < 2:
            with pytest.raises(ValueError):
                _val_X(_X, _intx_only)
        elif not _intx_only and _columns < 1:
            with pytest.raises(ValueError):
                _val_X(_X, _intx_only)
        else:
            assert _val_X(_X, _intx_only) is None














