# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from pybear.preprocessing.ColumnDeduplicateTransformer._validation. \
    _do_not_drop import _val_do_not_drop

import numpy as np
import pandas as pd
from uuid import uuid4

import pytest



class Fixtures:

    @staticmethod
    @pytest.fixture(scope='module')
    def _cols():
        return 5

    @staticmethod
    @pytest.fixture(scope='module')
    def X_np(_cols):
        return np.random.randint(0,10,(20,_cols))

    @staticmethod
    @pytest.fixture(scope='module')
    def good_columns(_cols):
        return [str(uuid4())[:4] for _ in range(_cols)]

    @staticmethod
    @pytest.fixture(scope='module')
    def X_pd(X_np, good_columns):
        return pd.DataFrame(data=X_np, columns=good_columns)



@pytest.mark.parametrize('_type', ('np', 'pd'), scope='module')
@pytest.mark.parametrize('_columns_is_passed', (True, False), scope='module')
class TestDoNotDropJunk(Fixtures):

    # fixtures ** * ** * ** * ** *

    @staticmethod
    @pytest.fixture(scope='module')
    def _X(X_np, X_pd, _type):
        if _type == 'np':
            _X = X_np
        elif _type == 'pd':
            _X = X_pd

        return _X

    @staticmethod
    @pytest.fixture(scope='module')
    def _columns(_columns_is_passed, good_columns):
        return good_columns if _columns_is_passed else None


    # END fixtures ** * ** * ** * ** *

    @pytest.mark.parametrize('junk_dnd',
        (-1, 0, 1, 3.14, True, 'trash', {'a':1}, max, lambda x: x)
    )
    def test_rejects_not_list_like_or_none(self, _X, _columns, junk_dnd):

        if isinstance(_X, pd.core.frame.DataFrame) and _columns is None:
            pytest.skip(reason=f"algorithmic impossibility")

        with pytest.raises(TypeError):
            _val_do_not_drop(junk_dnd, _X, _columns)


    @pytest.mark.parametrize('bad_dnd',
        (
            [True, min, 3.14],
            [min, max, float],
            [2.718, 3.141, 8.834]
        )
    )
    def test_rejects_bad_list(self, _X, _columns, bad_dnd):

        if isinstance(_X, pd.core.frame.DataFrame) and _columns is None:
            pytest.skip(reason=f"algorithmic impossibility")

        with pytest.raises(TypeError):
            _val_do_not_drop(bad_dnd, _X, _columns)


class TestDoNotDropArray(Fixtures):

    def test_array_rejects_str_when_columns_is_none(self, X_np, good_columns):

        with pytest.raises(TypeError):
            _val_do_not_drop(
                [v for i,v in enumerate(good_columns) if i%2==0],
                X_np,
                None
            )

    def test_array_accepts_good_str_when_columns_not_none(
        self, X_np, good_columns
    ):
        _val_do_not_drop(
            [v for i, v in enumerate(good_columns) if i % 2 == 0],
            X_np,
            good_columns
        )


    def test_array_rejects_bad_str_when_columns_not_none(
        self, X_np, good_columns
    ):

        with pytest.raises(TypeError):
            _val_do_not_drop(
                ['a', 'b'],
                X_np,
                None
            )


    @pytest.mark.parametrize('_columns_is_passed', (True, False))
    def test_array_accepts_good_int_always(
        self, X_np, _columns_is_passed, good_columns
    ):
        _val_do_not_drop(
            [0, 1],
            X_np,
            good_columns if _columns_is_passed else None
        )


    @pytest.mark.parametrize('_columns_is_passed', (True, False))
    def test_array_rejects_bad_int_always(
        self, X_np, _columns_is_passed, good_columns
    ):

        with pytest.raises(ValueError):
            _val_do_not_drop(
                [-1, 1],
                X_np,
                good_columns if _columns_is_passed else None
            )

        with pytest.raises(ValueError):
            _val_do_not_drop(
                [0, X_np.shape[1]],
                X_np,
                good_columns if _columns_is_passed else None
            )


    @pytest.mark.parametrize('_columns_is_passed', (True, False))
    def test_array_accepts_none_always(
        self, X_np, _columns_is_passed, good_columns
    ):
        _val_do_not_drop(
            None,
            X_np,
            good_columns if _columns_is_passed else None
        )


class TestDoNotDropDF(Fixtures):

    def test_df_accepts_good_str_always(self, X_pd, good_columns):

        # _columns IS ALWAYS PASSED IF X IS DF!

        _val_do_not_drop(
            [v for i, v in enumerate(good_columns) if i % 2 == 0],
            X_pd,
            good_columns
        )


    def test_df_rejects_bad_str_always(self, X_pd, good_columns):

        # _columns IS ALWAYS PASSED IF X IS DF!

        with pytest.raises(ValueError):
            _val_do_not_drop(
                ['a', 'b'],
                X_pd,
                good_columns
            )


    def test_df_accepts_good_int_always(self, X_pd, good_columns):

        # _columns IS ALWAYS PASSED IF X IS DF!

        _val_do_not_drop(
            [0, 1],
            X_pd,
            good_columns
        )


    def test_df_rejects_bad_int_always(self, X_pd, good_columns):

        # _columns IS ALWAYS PASSED IF X IS DF!

        with pytest.raises(ValueError):
            _val_do_not_drop(
                [-1, 1],
                X_pd,
                good_columns
            )

        with pytest.raises(ValueError):
            _val_do_not_drop(
                [0, X_pd.shape[1]],
                X_pd,
                good_columns
            )


    def test_columns_cannot_none_if_df(self, X_pd):
        with pytest.raises(ValueError):
            _val_do_not_drop(
                None,
                X_pd,
                None
            )



