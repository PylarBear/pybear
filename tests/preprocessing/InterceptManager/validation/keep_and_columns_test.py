# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.preprocessing.InterceptManager._validation._keep_and_columns import \
    _val_keep_and_columns


from copy import deepcopy
from uuid import uuid4
import numpy as np

import pytest





@pytest.fixture(scope='module')
def _shape():
    return (20, 3)


@pytest.fixture(scope='module')
def _good_columns(_shape):
    return [str(uuid4())[:4] for _ in range(_shape[1])]


@pytest.fixture(scope='module')
def _X(_X_factory, _shape):
    return _X_factory(
        _dupl=[[0, _shape[1]-1]],
        _has_nan=False,
        _constants={1: 1},
        _format='np',
        _columns=None,
        _dtype='flt',
        _zeros=None,
        _noise=float(0),
        _shape=_shape
    )


class TestValKeepAndColumns:

    # def _val_keep_and_columns(
    #     _keep:KeepType,
    #     _columns:Union[Iterable[str], None],
    #     _X: DataType
    # ) -> None:


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
        _val_keep_and_columns('first', good_columns, _X)


    @pytest.mark.parametrize('junk_keep',
        (3.14, True, False, None, [0,1], {0,1})
)
    def test_keep_rejects_junk(self, junk_keep, _good_columns, _X):
        # not int, str, dict[str, any], callable
        with pytest.raises(ValueError):
            _val_keep_and_columns(junk_keep, _good_columns, _X)


    @pytest.mark.parametrize('bad_keep',
        (-1, 999, {0:1}, {0:'junk'}, lambda x: 'trash', lambda x: -1)
    )
    def test_keep_rejects_bad(self, bad_keep, _good_columns, _X):
        # not int, str, dict[str, any], callable
        # negative int, int out of range, dict with non-str key,
        # callable returns bad index
        with pytest.raises(ValueError):
            _val_keep_and_columns(bad_keep, _good_columns, _X)


    @pytest.mark.parametrize('good_keep',
        (0, 1, 'first', 'last', 'random', 'none', {'Intercept': 1}, lambda x: 0)
    )
    def test_keep_accepts_good(self, good_keep, _good_columns, _X):
        # int, str, dict[str, any], callable that returns int >= 0
        _val_keep_and_columns(good_keep, _good_columns, _X)


    @pytest.mark.parametrize('_keep', ('first', 'last', 'random', 'none'))
    @pytest.mark.parametrize('conflict', (True, False))
    def test_raise_on_conflict_with_keep_literal(
        self, _keep, conflict, _good_columns, _X
    ):

        _columns = deepcopy(_good_columns)

        if conflict:
            _columns[np.random.randint(0, _X.shape[1])] = _keep

            with pytest.raises(ValueError):
                _val_keep_and_columns(_keep, _columns, _X)
        else:
            _val_keep_and_columns(_keep, _columns, _X)


    def test_rejects_non_literal_str_keep_with_no_header(self, _X):
        with pytest.raises(ValueError):
            _val_keep_and_columns('Some Column', None, _X)


    def test_rejects_non_literal_str_keep_not_in_header(self, _X, _good_columns):
        with pytest.raises(ValueError):
            _val_keep_and_columns('Some Column', _good_columns, _X)

