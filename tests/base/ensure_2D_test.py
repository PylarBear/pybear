# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.base._ensure_2D import ensure_2D

import numpy as np
import dask.array as da
import pandas as pd
import dask.dataframe as ddf
import dask_expr._collection as ddf2
import scipy.sparse as ss

import pytest




class TestEnsure2D:


    @pytest.mark.parametrize('junk_object',
        (-2.7, -1, 0, 1, 2.7, True, None, 'trash', {'a': 1}, lambda x: x, min)
    )
    def test_rejects_non_array_like(self, junk_object):
        with pytest.raises(ValueError):
            ensure_2D(junk_object)


    def test_rejects_does_not_have_shape_attr(self):

        with pytest.raises(ValueError):
            ensure_2D(
                [[0,1,2], [3,4,5], [6,7,8]]
            )

        with pytest.raises(ValueError):
            ensure_2D(
                ((0,1,2), (3,4,5), (6,7,8))
            )


    @pytest.mark.parametrize('X_format', ('np', 'pd', 'csr', 'da', 'ddf'))
    def test_accepts_array_like(self, X_format):

        _base_X = np.random.randint(0, 10, (10, 5))

        if X_format == 'np':
            _X = _base_X
        elif X_format == 'pd':
            _X = pd.DataFrame(data=_base_X)
        elif X_format == 'csr':
            _X = ss.csr_array(_base_X)
        elif X_format == 'da':
            _X = da.array(_base_X)
        elif X_format == 'ddf':
            _X = ddf.from_dask_array(da.array(_base_X))
        else:
            raise Exception


        ensure_2D(_X)


    @pytest.mark.parametrize('dim', (0, 3, 4))
    def test_blocks_0_dim_and_3_or_more_dim(self, dim):

        # build shape tuple
        _shape = tuple(np.random.randint(2, 5, dim).tolist())

        _X = np.random.randint(0, 10, _shape)

        with pytest.raises(ValueError):
            ensure_2D(_X)


    @pytest.mark.parametrize('X_format', ('np', 'pd', 'csr', 'da', 'ddf'))
    @pytest.mark.parametrize('dim', (1, 2))
    def test_accuracy(self, X_format, dim):

        # skip impossible conditions - - - - - - - - - - - - - - - - - -
        if X_format == 'csr' and dim == 1:
            pytest.skip(f"scipy sparse can only be 2D")
        # END skip impossible conditions - - - - - - - - - - - - - - - -


        # stay on the rails
        if dim not in [1,2]:
            raise Exception


        # build shape tuple
        _shape = tuple(np.random.randint(2, 10, dim).tolist())

        _base_X = np.random.randint(0, 10, _shape)

        if X_format == 'np':
            _X = _base_X
        elif X_format == 'pd':
            if dim == 1:
                _X = pd.Series(data=_base_X)
            elif dim == 2:
                _X = pd.DataFrame(data=_base_X)
        elif X_format == 'csr':
            _X = ss.csr_array(_base_X)
        elif X_format == 'da':
            _X = da.array(_base_X)
        elif X_format == 'ddf':
            _X = ddf.from_dask_array(da.array(_base_X))
            if dim == 1:
                X = _X.squeeze()
        else:
            raise Exception


        out = ensure_2D(_X)

        if X_format == 'pd':
            # anything 2D in pandas is always DF
            assert isinstance(out, pd.core.frame.DataFrame)
        elif X_format == 'ddf':
            # anything 2D in dask dataframe is always DF
            assert isinstance(out, (ddf.core.DataFrame, ddf2.DataFrame))
        else:
            assert type(out) == type(_X)

        assert len(out.shape) == 2

        # v v v v v verify data is in OBJECT is unchanged v v v v v v

        # convert out to np for array_equal
        if X_format == 'np':
            pass
        elif X_format == 'pd':
            out = out.to_numpy()
        elif X_format == 'csr':
            out = out.toarray()
        elif X_format == 'da':
            out = out.compute()
        elif X_format == 'ddf':
            out = out.compute().to_numpy()

        assert isinstance(out, np.ndarray)

        if dim == 1:
            assert np.array_equiv(out.ravel(), _base_X)
        elif dim == 2:
            assert np.array_equiv(out, _base_X)




