# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

import numpy as np
import pandas as pd
import polars as pl

from pybear.feature_extraction.text._TextReplacer._validation._X import _val_X



class TestValX:


    @pytest.mark.parametrize('junk_X',
        (-2.7, -1, 0, 1, 2.7, True, False, None, 'trash', {'A':1}, lambda x: x)
    )
    def test_rejects_non_list_like(self, junk_X):

        with pytest.raises(TypeError):

            _val_X(junk_X)


    @pytest.mark.parametrize('dim', (1, 2))
    @pytest.mark.parametrize('container',
        (list, set, tuple, np.ndarray, pd.Series, pd.DataFrame, pl.DataFrame)
    )
    def test_rejects_list_like_non_str(self, dim, container):

        if dim == 1 and container is pd.DataFrame:
            pytest.skip(reason=f'impossible condition')

        if dim == 1 and container is pl.DataFrame:
            pytest.skip(reason=f'impossible condition')

        if dim == 2 and container is pd.Series:
            pytest.skip(reason=f'impossible condition')

        if dim == 2 and container is set:
            pytest.skip(reason=f'impossible condition')

        # END skip impossible conditions -- -- -- -- -- -- -- -- -- --

        if dim == 1:

            _X = np.random.randint(0, 10, (10,))

            if container is np.ndarray:
                assert isinstance(_X, np.ndarray)
            elif container is pd.Series:
                _X = pd.Series(_X)
                assert isinstance(_X, pd.Series)
            elif container is pl.Series:
                _X = pl.Series(_X)
                assert isinstance(_X, pl.Series)
            else:  # list, set, tuple
                _X = container(_X.tolist())
                assert isinstance(_X, container)

        elif dim == 2:

            _X = np.random.randint(0, 10, (20, 10))

            if container is np.ndarray:
                assert isinstance(_X, np.ndarray)
            elif container is pd.DataFrame:
                _X = pd.DataFrame(_X)
                assert isinstance(_X, pd.DataFrame)
            elif container is pl.DataFrame:
                _X = pl.DataFrame(_X)
                assert isinstance(_X, pl.DataFrame)
            elif container is list:
                _X = list(map(list, _X))
                assert isinstance(_X, list)
            elif container is tuple:
                _X = tuple(map(tuple, _X))
                assert isinstance(_X, tuple)
            else:
                raise Exception

        with pytest.raises(TypeError):

            _val_X(_X)



    @pytest.mark.parametrize('dim', (1, 2))
    @pytest.mark.parametrize('container',
        (list, set, tuple, np.ndarray, pd.Series, pd.DataFrame, pl.DataFrame)
    )
    def test_accepts_list_like_str(self, dim, container):


        if dim == 1 and container is pd.DataFrame:
            pytest.skip(reason=f'impossible condition')

        if dim == 1 and container is pl.DataFrame:
            pytest.skip(reason=f'impossible condition')

        if dim == 2 and container is pd.Series:
            pytest.skip(reason=f'impossible condition')

        if dim == 2 and container is set:
            pytest.skip(reason=f'impossible condition')

        # END skip impossible conditions -- -- -- -- -- -- -- -- -- --

        if dim == 1:

            _X = np.random.choice(list('abcde'), (10,), replace=True)

            if container is np.ndarray:
                assert isinstance(_X, np.ndarray)
            elif container is pd.Series:
                _X = pd.Series(_X)
                assert isinstance(_X, pd.Series)
            elif container is pl.Series:
                _X = pl.Series(_X)
                assert isinstance(_X, pl.Series)
            else:  # list, set, tuple
                _X = container(_X.tolist())
                assert isinstance(_X, container)

        elif dim == 2:

            _X = np.random.choice(list('abcde'), (20,10), replace=True)

            # make python objects ragged so we know raggedness is handled

            if container is np.ndarray:
                assert isinstance(_X, np.ndarray)
            elif container is pd.DataFrame:
                _X = pd.DataFrame(_X)
                assert isinstance(_X, pd.core.frame.DataFrame)
            elif container is pl.DataFrame:
                _X = pl.DataFrame(_X)
                assert isinstance(_X, pl.DataFrame)
            elif container is list:
                _X = [
                    list('abc'),
                    list('abcde'),
                    list('ab'),
                    list('abcd')
                ]
                assert isinstance(_X, list)
            elif container is tuple:
                _X = tuple([
                    tuple('abc'),
                    tuple('abcde'),
                    tuple('ab'),
                    tuple('abcd')
                ])
                assert isinstance(_X, tuple)
            else:
                raise Exception


        if isinstance(_X, pd.core.frame.DataFrame):
            with pytest.raises(TypeError):
                _val_X(_X)
        else:
            assert _val_X(_X) is None



