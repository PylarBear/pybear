# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.preprocessing.InterceptManager._shared._manage_keep import (
    _manage_keep
)

import numpy as np
import pandas as pd

import pytest


# no scipy sparse are tested here, should be comparable to numpy

class TestManageKeep:

    # def _manage_keep(
    #     _keep: KeepType,
    #     _X: DataFormatType,
    #     constant_columns_: dict[int, any],
    #     _n_features_in: int,
    #     _feature_names_in: Union[npt.NDArray[str], None]
    # ) -> Union[Literal['first', 'last', 'random', 'none'], dict[str, any], int]:


    # callable keep converts X to int, validated against constant_columns_
    # keep feature str converted to int, validated against constant_columns_
    # int keep validated against constant_columns_
    # keep in ('first', 'last', 'random') warns if no constants, otherwise
    #   converted to int
    # keep == 'none', passes through
    # isinstance(_keep, dict), passes through


    @staticmethod
    @pytest.fixture(scope='module')
    def _shape():
        return (20, 10)


    @staticmethod
    @pytest.fixture(scope='module')
    def _X_np(_shape):
        return np.random.randint(0, 10, _shape)


    @staticmethod
    @pytest.fixture(scope='module')
    def _X_pd(_X_np, _columns):
        return pd.DataFrame(data=_X_np, columns=_columns)


    @staticmethod
    @pytest.fixture(scope='module')
    def _columns(_master_columns, _shape):
        return _master_columns.copy()[:_shape[1]]


    # dict v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^
    @pytest.mark.parametrize('_format', ('np', 'pd'))
    @pytest.mark.parametrize('_keep',
        ({'Intercept': 1}, {'innards': 'not validated'})
    )
    @pytest.mark.parametrize('_constant_columns', ({}, {0:1, 1:np.nan, 2:0}))
    def test_dict_passes_thru(
        self, _X_np, _X_pd, _format, _keep, _constant_columns, _shape
    ):

        # dict passes thru. len==1 & key is str validated in _keep_and_columns
        if _format == 'pd':
            _X = _X_pd
            _columns = _X_pd.columns.to_numpy()
        else:
            _X = _X_np
            _columns = None

        out = _manage_keep(
            _keep=_keep,
            _X=_X,
            constant_columns_=_constant_columns,
            _n_features_in=_shape[1],
            _feature_names_in=_columns
        )

        assert isinstance(out, dict), "_manage_keep dict did not return dict"
        assert out == _keep, f"_manage_keep altered keep dict[str, any]"
    # END dict v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^


    # callable v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^
    @pytest.mark.parametrize('_format', ('np', 'pd'))
    @pytest.mark.parametrize('_keep', (lambda x: 0, lambda x: 100))
    @pytest.mark.parametrize('_constant_columns', ({}, {0:1, 1:np.nan, 2:0}))
    def test_callable(
        self, _X_np, _X_pd, _format, _keep, _constant_columns, _shape
    ):

        # from _keep_and_columns, we already know that keep callable returns an
        # integer within range of X.shape[1]. just needs to verify the
        # returned idx is actually a column of constants.

        if _format == 'pd':
            _X = _X_pd
            _columns = _X.columns.to_numpy()
        else:
            _X = _X_np
            _columns = None

        if _keep(_X) in _constant_columns:
            out = _manage_keep(
                _keep=_keep,
                _X=_X,
                constant_columns_=_constant_columns,
                _n_features_in=_shape[1],
                _feature_names_in=_columns
            )

            assert isinstance(out, int), \
                f"_manage_keep callable did not return integer"
            assert out == _keep(_X), \
                (f"_manage_keep did not return expected keep callable output. "
                 f"exp {_keep(_X)}, got {out}")

        elif _keep(_X) not in _constant_columns:
            with pytest.raises(ValueError):
                _manage_keep(
                    _keep=_keep,
                    _X=_X,
                    constant_columns_=_constant_columns,
                    _n_features_in=_shape[1],
                    _feature_names_in=_columns
                )
    # END callable v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^

    # feature name str v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^
    @pytest.mark.parametrize('_keep', ('column_1', 'column_2', 'column_3'))
    @pytest.mark.parametrize('_constant_columns', ({}, {0: 1, 1: np.nan}))
    def test_feature_name_str(
        self, _X_pd, _keep, _columns, _constant_columns, _shape
    ):

        # _keep_and_columns caught if keep feature name:
        # - was passed with no header (ie for a numpy array) -
        #       dont need to test np, impossible condition.
        # - is not in the header - dont need to test, impossible condition

        # _manage_keep only validates if __keep not in _constant_columns

        exp = {'column_1': 0, 'column_2': 1, 'column_3': 2}[_keep]

        if _keep == 'column_1':
            _keep = _columns[0]
        elif _keep == 'column_2':
            _keep = _columns[1]
        elif _keep == 'column_3':
            _keep = _columns[2]
        else:
            raise Exception

        if exp in _constant_columns:
            out = _manage_keep(
                _keep=_keep,
                _X=_X_pd,
                constant_columns_=_constant_columns,
                _n_features_in=_shape[1],
                _feature_names_in=_X_pd.columns.to_numpy()
            )

            assert isinstance(out, int), \
                f"_manage_keep feature name did not return integer"
            assert out == exp, \
                (f"_manage_keep did not return expected keep feature name index "
                 f"exp {exp}, got {out}")

        elif exp not in _constant_columns:
            with pytest.raises(ValueError):
                _manage_keep(
                    _keep=_keep,
                    _X=_X_pd,
                    constant_columns_=_constant_columns,
                    _n_features_in=_shape[1],
                    _feature_names_in=_X_pd.columns.to_numpy()
                )
    # END feature name str v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^


    # literal str v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v
    @pytest.mark.parametrize('_format', ('np', 'pd'))
    @pytest.mark.parametrize('_keep', ('first', 'last', 'random', 'none'))
    @pytest.mark.parametrize('_constant_columns',
        ({}, {0: 1, 1: np.nan, 2: 0}, {7:1, 8:0, 9:np.e}, {4:-1, 3:2})
    )
    def test_literal_str(
        self, _X_np, _X_pd, _columns, _format, _keep, _constant_columns, _shape
    ):

        # what we know:
        # the only possible strings that can get to _manage_keep besides
        # feature names are literals ('first', 'last', 'random', 'none')
        # any other string would except in _keep_and_columns
        # only need to test the exact cases of the literals
        # unless 'keep' is 'none' or constant_columns_ is empty, the
        # returned value must be in constant_columns_. if constant_columns_
        # is empty, returns 'none'

        if _format == 'pd':
            _X = _X_pd
            _columns = _X.columns.to_numpy()
        else:
            _X = _X_np
            _columns = None


        out = _manage_keep(
            _keep=_keep,
            _X=_X,
            constant_columns_=_constant_columns,
            _n_features_in=_shape[1],
            _feature_names_in=_columns
        )

        _sorted_const_cols = sorted(list(_constant_columns.keys()))
        if len(_sorted_const_cols) == 0:
            exp = 'none'
        elif _keep == 'first':
            exp = _sorted_const_cols[0]
        elif _keep == 'last':
            exp = _sorted_const_cols[-1]
        elif _keep == 'random':
            exp = 'random'
        elif _keep == 'none':
            exp = 'none'
        else:
            raise Exception

        if len(_constant_columns) == 0:
            assert isinstance(out, str), \
                f"_manage_keep literal did not return str: {out}"
        elif _keep in ('first', 'last', 'random'):
            assert isinstance(out, int), \
                f"_manage_keep literal did not return integer: {out}"
        elif _keep == 'none':
            assert isinstance(out, str), \
                f"_manage_keep literal did not return str: {out}"
        else:
            raise Exception

        if _keep in ('first', 'last', 'none') or len(_constant_columns) == 0:
            assert out == exp, \
                (f"_manage_keep did not return expected keep literal "
                 f"index, exp {exp}, got {out}")
        elif _keep == 'random':
            assert out in range(_shape[1])
    # END literal str v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v

    # int v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v

    @pytest.mark.parametrize('_format', ('np', 'pd'))
    @pytest.mark.parametrize('_keep', (0, 10, 2000, 1_000_000_000))
    @pytest.mark.parametrize('_constant_columns', ({}, {0:1, 1:np.nan, 9:0}))
    def test_integer(
        self, _X_np, _X_pd, _format, _keep, _constant_columns, _shape
    ):

        # from _keep_and_columns, we already know that keep integer is in
        # range(X.shape[1]. just need to verify the idx is actually a
        # column of constants.

        if _format == 'pd':
            _X = _X_pd
            _columns = _X.columns.to_numpy()
        else:
            _X = _X_np
            _columns = None

        if _keep in _constant_columns:
            out = _manage_keep(
                _keep=_keep,
                _X=_X,
                constant_columns_=_constant_columns,
                _n_features_in=_shape[1],
                _feature_names_in=_columns
            )

            assert isinstance(out, int), \
                f"_manage_keep integer did not return integer"
            assert out == _keep, \
                (f"_manage_keep integer did not return expected integer. "
                 f"exp {_keep}, got {out}")

        elif _keep not in _constant_columns:
            with pytest.raises(ValueError):
                _manage_keep(
                    _keep=_keep,
                    _X=_X,
                    constant_columns_=_constant_columns,
                    _n_features_in=_shape[1],
                    _feature_names_in=_columns
                )
    # END int v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v


















