# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.preprocessing.InterceptManager._partial_fit._make_instructions \
    import _make_instructions

import numpy as np

import pytest



class TestMakeInstructions:

    # def _make_instructions(
    #     _keep: KeepType,
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


    @staticmethod
    @pytest.fixture(scope='module')
    def _keep_str(_columns):
        return _columns[8]    # not arbitrary, must be index 8


    @staticmethod
    @pytest.fixture(scope='module')
    def _keep_callable(_columns):
        return lambda x: 8  # not arbitrary, must equal 8


    def test_accuracy(
        self, _empty_constant_columns, _constant_columns_1, _constant_columns_2,
        _keep_dict, _keep_int, _keep_str, _keep_callable, _columns, _shape
    ):


        # pizza u need to add tests for keep is int, str, and callable.
        # just too tired 24_11_15_16_58_00

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
        # keep is str ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
        # if no constant columns, returns all Nones
        out = _make_instructions(_keep_str, _empty_constant_columns, _columns, _shape)
        assert out == {'keep': None, 'delete': None, 'add': None}

        # keep first constant column, delete all others
        out = _make_instructions(_keep_str, _constant_columns_1, _columns, _shape)
        _sorted = sorted(list(_constant_columns_1))
        _kept_idx = np.arange(len(_columns))[_columns==_keep_str][0]
        _sorted.remove(_kept_idx)
        assert out == {'keep': [_kept_idx], 'delete': _sorted, 'add': None}
        del _sorted, _kept_idx

        # keep first constant column, delete all others
        out = _make_instructions(_keep_str, _constant_columns_2, _columns, _shape)
        _sorted = sorted(list(_constant_columns_2))
        _kept_idx = np.where(_columns==_keep_str)[0]
        _sorted.remove(_kept_idx)
        assert out == {'keep': [_kept_idx], 'delete': _sorted, 'add': None}
        del _sorted, _kept_idx
        # END keep is str ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
        # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

        # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
        # keep is callable ** * ** * ** * ** * ** * ** * ** * ** * ** * **
        # if no constant columns, returns all Nones
        out = _make_instructions(_keep_callable, _empty_constant_columns, None, _shape)
        assert out == {'keep': None, 'delete': None, 'add': None}

        # keep first constant column, delete all others
        with pytest.raises(ValueError):
            out = _make_instructions(_keep_callable, _constant_columns_1, None, _shape)
        # pizza
        # _sorted = sorted(list(_constant_columns_1))
        # _kept_idx = _keep_callable(np.random.randint(0,10,(5,3)))  # the callable isnt validated, could pass anything
        # _sorted.remove(_kept_idx)
        # assert out == {'keep': [_kept_idx], 'delete': _sorted, 'add': None}
        # del _sorted, _kept_idx

        # keep first constant column, delete all others
        with pytest.raises(ValueError):
            out = _make_instructions(_keep_callable, _constant_columns_2, None, _shape)
        # pizza
        # _sorted = sorted(list(_constant_columns_2))
        # _kept_idx = _keep_callable(np.random.randint(0, 10, (5, 3)))  # the callable isnt validated, could pass anything
        # _sorted.remove(_kept_idx)
        # assert out == {'keep': [_kept_idx], 'delete': _sorted, 'add': None}
        # del _sorted, _kept_idx
        # END keep is callable ** * ** * ** * ** * ** * ** * ** * ** * ** *
        # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

        # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
        # first ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
        # if no constant columns, returns all Nones
        out = _make_instructions('first', _empty_constant_columns, None, _shape)
        assert out == {'keep': None, 'delete': None, 'add': None}

        # keep first constant column, delete all others
        out = _make_instructions('first', _constant_columns_1, None, _shape)
        _sorted = sorted(list(_constant_columns_1))
        _kept_idx = _sorted[0]
        _sorted.remove(_kept_idx)
        assert out == {'keep': [_kept_idx], 'delete': _sorted, 'add': None}
        del _sorted, _kept_idx

        # keep first constant column, delete all others
        out = _make_instructions('first', _constant_columns_2, None, _shape)
        _sorted = sorted(list(_constant_columns_2))
        _kept_idx = _sorted[0]
        _sorted.remove(_kept_idx)
        assert out == {'keep': [_kept_idx], 'delete': _sorted, 'add': None}
        del _sorted, _kept_idx
        # END first ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
        # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

        # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
        # last ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
        # if no constant columns, returns all Nones
        out = _make_instructions('last', _empty_constant_columns, None, _shape)
        assert out == {'keep': None, 'delete': None, 'add': None}

        # keep last constant column, delete all others
        out = _make_instructions('last', _constant_columns_1, None, _shape)
        _sorted = sorted(list(_constant_columns_1))
        _kept_idx = _sorted[-1]
        _sorted.remove(_kept_idx)
        assert out == {'keep': [_kept_idx], 'delete': _sorted, 'add': None}
        del _sorted, _kept_idx

        # keep last constant column, delete all others
        out = _make_instructions('last', _constant_columns_2, None, _shape)
        _sorted = sorted(list(_constant_columns_2))
        _kept_idx = _sorted[-1]
        _sorted.remove(_kept_idx)
        assert out == {'keep': [_kept_idx], 'delete': _sorted, 'add': None}
        del _sorted, _kept_idx
        # END last ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
        # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

        # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
        # random ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
        # if no constant columns, returns all Nones
        out = _make_instructions('random', _empty_constant_columns, None, _shape)
        assert out == {'keep': None, 'delete': None, 'add': None}

        # keep a random constant column, delete all others
        out = _make_instructions('random', _constant_columns_1, None, _shape)
        # can only validate len of instructions
        assert len(out['keep']) == 1
        assert len(out['delete']) == len(_constant_columns_1) - 1
        assert out['add'] is None

        # keep a random constant column, delete all others
        out = _make_instructions('random', _constant_columns_2, None, _shape)
        # can only validate len of instructions
        assert len(out['keep']) == 1
        assert len(out['delete']) == len(_constant_columns_2) - 1
        assert out['add'] is None
        # END random ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
        # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

        # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
        # none ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
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
        # END none ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
        # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

        # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
        # dict ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
        # if no constant columns, returns all Nones
        out = _make_instructions(_keep_dict, _empty_constant_columns, None, _shape)
        assert out == {'keep': None, 'delete': None, 'add': None}

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










