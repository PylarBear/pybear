# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.preprocessing.InterceptManager._shared._make_instructions \
    import _make_instructions

import pytest



class TestMakeInstructions:

    # def _make_instructions(
    #     _keep: Union[int, Literal['none'], dict[str, any]],
    #     constant_columns_: dict[int, any]
    # ) -> InstructionType:


    @staticmethod
    @pytest.fixture(scope='module')
    def _shape():
        return (200, 10)   # arbitrary shape


    @staticmethod
    @pytest.fixture(scope='module')
    def _columns(_master_columns, _shape):
        return _master_columns.copy()[:_shape[1]]   # arbitrary shape


    @staticmethod
    @pytest.fixture(scope='module')
    def _empty_constant_columns():
        return {}


    @staticmethod
    @pytest.fixture(scope='module')
    def _constant_columns_1(_shape):
        _ = {0: 1, 8: 1}  # must have index 8 in it
        assert max(_) < _shape[1]
        return _


    @staticmethod
    @pytest.fixture(scope='module')
    def _constant_columns_2(_shape):
        _ = {1: 1, 0: 0, 8: 1}   # must have index 8 in it
        assert max(_) < _shape[1]
        return _


    @staticmethod
    @pytest.fixture(scope='module')
    def _keep_dict():
        return {'Intercept': 1}


    @staticmethod
    @pytest.fixture(scope='module')
    def _keep_int(_columns):
        return len(_columns) - 2    # not arbitrary, must equal 8


    def test_accuracy(
        self, _empty_constant_columns, _constant_columns_1, _constant_columns_2,
        _keep_dict, _keep_int, _columns, _shape
    ):

        # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
        # keep is int ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
        # if no constant columns, returns all Nones
        out = _make_instructions(_keep_int, _empty_constant_columns, None, _shape)
        assert out == {'keep': None, 'delete': None, 'add': None}

        # keep _keep_int idx, delete all others
        out = _make_instructions(_keep_int, _constant_columns_1, None, _shape)
        _sorted = sorted(list(_constant_columns_1))
        _kept_idx = _keep_int
        _sorted.remove(_kept_idx)
        assert out == {'keep': [_kept_idx], 'delete': _sorted, 'add': None}
        del _sorted, _kept_idx

        # keep _keep_int idx, delete all others
        out = _make_instructions(_keep_int, _constant_columns_2, None, _shape)
        _sorted = sorted(list(_constant_columns_2))
        _kept_idx = _keep_int
        _sorted.remove(_kept_idx)
        assert out == {'keep': [_kept_idx], 'delete': _sorted, 'add': None}
        del _sorted, _kept_idx
        # END keep is int ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
        # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

        # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
        # 'none' ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
        # if no constant columns, returns all Nones
        out = _make_instructions('none', _empty_constant_columns, None, _shape)
        assert out == {'keep': None, 'delete': None, 'add': None}

        # delete all constant columns
        out = _make_instructions('none', _constant_columns_1, None, _shape)
        assert out == {
            'keep': None,
            'delete': sorted(list(_constant_columns_1)),
            'add': None
        }

        # delete all constant columns
        out = _make_instructions('none', _constant_columns_2, None, _shape)
        assert out == {
            'keep': None,
            'delete': sorted(list(_constant_columns_2)),
            'add': None
        }
        # END 'none' ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
        # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

        # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
        # dict ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
        # if no constant columns, returns all Nones except 'add'
        out = _make_instructions(_keep_dict, _empty_constant_columns, None, _shape)
        assert out == {'keep': None, 'delete': None, 'add': _keep_dict}

        # delete all constant columns, append contents of keep dict
        out = _make_instructions(_keep_dict, _constant_columns_1, None, _shape)
        assert out == {
            'keep': None,
            'delete': sorted(list(_constant_columns_1)),
            'add': _keep_dict
        }

        # delete all constant columns, append contents of keep dict
        out = _make_instructions(_keep_dict, _constant_columns_2, None, _shape)
        assert out == {
            'keep': None,
            'delete': sorted(list(_constant_columns_2)),
            'add': _keep_dict
        }
        # END dict ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
        # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    def test_all_columns_constant(self):

        # if all columns are constant and not appending new constants, raise
        with pytest.raises(ValueError):
            _make_instructions(
                _keep='none',
                constant_columns_=dict((zip(range(5), (1 for _ in range(5))))),
                _columns=None,
                _shape=(1_000_000_000, 5)
            )

        # if all columns are constant but appending new constants, warn
        with pytest.warns():
            out = _make_instructions(
                _keep={'Intercept': 1},
                constant_columns_=dict((zip(range(5), (1 for _ in range(5))))),
                _columns=None,
                _shape=(1_000_000_000, 5)
            )

        assert out['keep'] == None
        assert out['delete'] == list(range(5))
        assert out['add'] == {'Intercept': 1}



