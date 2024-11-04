# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from pybear.preprocessing.NoDupPolyFeatures._validation. \
    _do_not_drop import _val_do_not_drop

import numpy as np
import pandas as pd


import pytest



class Fixtures:

    @staticmethod
    @pytest.fixture(scope='module')
    def _shape():
        return (20, 5)

    @staticmethod
    @pytest.fixture(scope='module')
    def _columns(_master_columns, _shape):
        return _master_columns.copy()[:_shape[1]]

    @staticmethod
    @pytest.fixture(scope='module')
    def X_np(_X_factory, _shape):
        return _X_factory(_format='np', _shape=_shape)

    @staticmethod
    @pytest.fixture(scope='module')
    def X_pd(X_np, _columns):
        return pd.DataFrame(data=X_np, columns=_columns)



@pytest.mark.parametrize('_type', ('np', 'pd'), scope='module')
@pytest.mark.parametrize('_columns_is_passed', (True, False), scope='module')
class TestDoNotDropJunk(Fixtures):


    @pytest.mark.parametrize('junk_dnd',
        (-1, 0, 1, 3.14, True, 'trash', {'a':1}, max, lambda x: x)
    )
    def test_rejects_not_list_like_or_none(
        self, X_np, X_pd, _type, _columns, _columns_is_passed, junk_dnd):

        _X = X_np if _type == 'np' else X_pd

        if isinstance(_X, pd.core.frame.DataFrame) and not _columns_is_passed:
            pytest.skip(reason=f"algorithmic impossibility")

        with pytest.raises(TypeError):
            _val_do_not_drop(
                junk_dnd,
                _X,
                _columns if _columns_is_passed else None
            )


    @pytest.mark.parametrize('bad_dnd',
        (
            [True, min, 3.14],
            [min, max, float],
            [2.718, 3.141, 8.834],
            []
        )
    )
    def test_rejects_bad_list(
        self, X_np, X_pd, _type, _columns, _columns_is_passed, bad_dnd
    ):

        _X = X_np if _type == 'np' else X_pd

        if isinstance(_X, pd.core.frame.DataFrame) and not _columns_is_passed:
            pytest.skip(reason=f"algorithmic impossibility")

        with pytest.raises(TypeError):
            _val_do_not_drop(
                bad_dnd,
                _X,
                _columns if _columns_is_passed else None
            )


class TestDoNotDropArray(Fixtures):

    def test_array_str_handing(self, X_np, _columns):

        # rejects str when columns is none
        with pytest.raises(TypeError):
            _val_do_not_drop(
                [v for i,v in enumerate(_columns) if i%2==0],
                X_np,
                None
            )

        # accepts good str when columns not none
        _val_do_not_drop(
            [v for i, v in enumerate(_columns) if i % 2 == 0],
            X_np,
            _columns
        )

        # rejects bad str when columns not none
        with pytest.raises(TypeError):
            _val_do_not_drop(
                ['a', 'b'],
                X_np,
                None
            )


    @pytest.mark.parametrize('_columns_is_passed', (True, False))
    def test_array_int_and_none_handling(
        self, X_np, _columns_is_passed, _columns
    ):

        # accepts good int always
        _val_do_not_drop(
            [0, 1],
            X_np,
            _columns if _columns_is_passed else None
        )

        # rejects bad int always - 1
        with pytest.raises(ValueError):
            _val_do_not_drop(
                [-1, 1],
                X_np,
                _columns if _columns_is_passed else None
            )

        # rejects bad int always - 2
        with pytest.raises(ValueError):
            _val_do_not_drop(
                [0, X_np.shape[1]],
                X_np,
                _columns if _columns_is_passed else None
            )

        # accepts None always
        _val_do_not_drop(
            None,
            X_np,
            _columns if _columns_is_passed else None
        )


class TestDoNotDropDF(Fixtures):

    # _columns IS ALWAYS PASSED IF X IS DF!

    def test_df_str_handling(self, X_pd, _columns):

        # accepts good str always
        _val_do_not_drop(
            [v for i, v in enumerate(_columns) if i % 2 == 0],
            X_pd,
            _columns
        )

        # rejects bad str always
        with pytest.raises(ValueError):
            _val_do_not_drop(
                ['a', 'b'],
                X_pd,
                _columns
            )


    def test_df_int_and_none_handling(self, X_pd, _columns):

        # accepts good int always
        _val_do_not_drop(
            [0, 1],
            X_pd,
            _columns
        )

        # rejects bad int always - 1
        with pytest.raises(ValueError):
            _val_do_not_drop(
                [-1, 1],
                X_pd,
                _columns
            )

        # rejects bad int always - 2
        with pytest.raises(ValueError):
            _val_do_not_drop(
                [0, X_pd.shape[1]],
                X_pd,
                _columns
            )


        # columns can be None
        _val_do_not_drop(
            None,
            X_pd,
            None
        )










