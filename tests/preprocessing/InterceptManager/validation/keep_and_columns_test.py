# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

from copy import deepcopy
import random

import numpy as np
import pandas as pd

from pybear.preprocessing._InterceptManager._validation._keep_and_columns import \
    _val_keep_and_columns



class TestValKeepAndColumns:

    # def _val_keep_and_columns(
    #     _keep:KeepType,
    #     _columns:Union[Sequence[str], None],
    #     _X: DataContainer
    # ) -> None:


    # fixtures ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

    @staticmethod
    @pytest.fixture(scope='module')
    def _X(_shape):
        return np.random.randint(0, 10, _shape)

    # END fixtures ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **


    @pytest.mark.parametrize(f'_junk_columns',
        (-1, 0, 1, 3.14, True, False, 'junk', {'a': 1}, lambda x: x)
    )
    def test_columns_rejects_junk(self, _junk_columns, _X):
        # not list-like
        with pytest.raises(ValueError):
            _val_keep_and_columns('first', _junk_columns, _X)


    @pytest.mark.parametrize(f'_bad_columns',
        ((1,2,3), {True, False}, ['too', 'short'], list('toolong'))
    )
    def test_columns_rejects_bad(self, _bad_columns, _X):
        # list-like, but not strs or too long or too short
        with pytest.raises(ValueError):
            _val_keep_and_columns('first', _bad_columns, _X)


    @pytest.mark.parametrize(f'good_columns',
        (None, tuple('abc'), list('123'), set('qrs'))
    )
    def test_columns_accepts_good(self, good_columns, _X):
        # None, or list like of strings with correct shape
        _val_keep_and_columns('first', good_columns, _X[:, :3])


    @pytest.mark.parametrize('junk_keep',
        (3.14, True, False, None, [0,1], {0,1})
    )
    def test_keep_rejects_junk(self, junk_keep, _columns, _X):
        # not int, str, dict[str, any], callable
        with pytest.raises(ValueError):
            _val_keep_and_columns(junk_keep, _columns, _X)


    @pytest.mark.parametrize('bad_keep',
        (-1, 999, {0:1}, {0:'junk'}, lambda x: 'trash', lambda x: -1)
    )
    def test_keep_rejects_bad(self, bad_keep, _columns, _X):
        # not int, str, dict[str, any], callable
        # negative int, int out of range, dict with non-str key,
        # callable returns bad index
        with pytest.raises(ValueError):
            _val_keep_and_columns(bad_keep, _columns, _X)


    @pytest.mark.parametrize('good_keep',
        (0, 1, 'first', 'last', 'random', 'none', {'Intercept': 1}, lambda x: 0)
    )
    def test_keep_accepts_good(self, good_keep, _columns, _X):
        # int, str, dict[str, any], callable that returns int >= 0
        _val_keep_and_columns(good_keep, _columns, _X)


    @pytest.mark.parametrize('_keep', ('first', 'last', 'random', 'none'))
    @pytest.mark.parametrize('conflict', (True, False))
    def test_raise_on_header_conflict_with_keep_literal(
        self, _keep, conflict, _columns, _X, _shape
    ):

        if conflict:
            _conflict_columns = deepcopy(_columns)
            _conflict_columns[random.choice(range(_shape[1]))] = _keep

            with pytest.raises(ValueError):
                _val_keep_and_columns(_keep, _conflict_columns, _X)
        else:
            _val_keep_and_columns(_keep, _columns, _X)


    def test_keep_rejects_non_literal_str_with_no_header(self, _X):
        with pytest.raises(ValueError):
            _val_keep_and_columns('Some Column', None, _X)


    def test_keep_rejects_non_literal_str_not_in_header(self, _X, _columns):
        with pytest.raises(ValueError):
            _val_keep_and_columns('Some Column', _columns, _X)


    def test_warns_if_keep_dict_key_in_columns(self, _X):

        X = pd.DataFrame(
            data=_X[:, :3],
            columns=['x1', 'x2', 'Intercept']
        )

        with pytest.warns():
            _val_keep_and_columns(
                _keep={'Intercept': 1},
                _columns=X.columns,
                _X=X
            )


    @pytest.mark.parametrize(f'keep_value', ([0, 1], {0,1}, (0,1), {'a':1}))
    def test_rejects_keep_dict_value_is_nonstr_sequence(self, _X, keep_value):
        # {'Intercept': value}, value cannot be list-like sequence

        with pytest.raises(ValueError):
            _val_keep_and_columns(
                _keep={'Intercept': keep_value},
                _columns=None,
                _X=_X
            )


    @pytest.mark.parametrize(f'keep_value', (min, max, lambda x: x, list))
    def test_reject_keep_dict_value_is_callable(self, _X, keep_value):
        # {'Intercept': value}, value cannot be callable

        with pytest.raises(ValueError):
            _val_keep_and_columns(
                _keep={'Intercept': keep_value},
                _columns=None,
                _X=_X
            )


    @pytest.mark.parametrize(f'keep_value',
        (-np.e, -1, 0, 1, np.e, True, False, np.nan, pd.NA, 'strings')
    )
    def test_accept_keep_dict_value(self, _X, keep_value):
        # {'Intercept': value}, value can be int, float, bool, str

        _val_keep_and_columns(
            _keep={'Intercept': keep_value},
            _columns=None,
            _X=_X
        )








