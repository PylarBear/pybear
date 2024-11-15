# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.preprocessing.InterceptManager._partial_fit._make_instructions \
    import _make_instructions


import pytest



class TestMakeInstructions:

    # def _make_instructions(
    #     _keep: KeepType,
    #     constant_columns_: dict[int, any]
    # ) -> InstructionType:


    @staticmethod
    @pytest.fixture(scope='module')
    def _empty_constant_columns():
        return {}


    @staticmethod
    @pytest.fixture(scope='module')
    def _constant_columns_1():
        return {0: 1, 13: 1}


    @staticmethod
    @pytest.fixture(scope='module')
    def _constant_columns_2():
        return {1: 1, 0: 0, 8: 1}


    @staticmethod
    @pytest.fixture(scope='module')
    def _keep_dict():
        return {'Intercept': 1}


    @pytest.mark.parametrize(f'_trial', ('trial_1', 'trial_2', 'trial_3'))
    def test_accuracy(
        self, _trial, _empty_constant_columns, _constant_columns_1,
        _constant_columns_2, _keep_dict
    ):

        # first
        if _trial == 'trial_1':
            # if no constant columns, returns all Nones
            out = _make_instructions('first', _empty_constant_columns)
            assert out == {'keep': None, 'delete': None, 'add': None}
        elif _trial == 'trial_2':
            # keep first constant column, delete all others
            out = _make_instructions('first', _constant_columns_1)
            _sorted = sorted(list(_constant_columns_1))
            _kept_idx = _sorted[0]
            _sorted.remove(_kept_idx)
            assert out == {'keep': [_kept_idx], 'delete': _sorted, 'add': None}
            del _sorted, _kept_idx
        elif _trial == 'trial_3':
            # keep first constant column, delete all others
            out = _make_instructions('first', _constant_columns_2)
            _sorted = sorted(list(_constant_columns_2))
            _kept_idx = _sorted[0]
            _sorted.remove(_kept_idx)
            assert out == {'keep': [_kept_idx], 'delete': _sorted, 'add': None}
            del _sorted, _kept_idx

        # last
        if _trial == 'trial_1':
            # if no constant columns, returns all Nones
            out = _make_instructions('last', _empty_constant_columns)
            assert out == {'keep': None, 'delete': None, 'add': None}
        elif _trial == 'trial_2':
            # keep last constant column, delete all others
            out = _make_instructions('last', _constant_columns_1)
            _sorted = sorted(list(_constant_columns_1))
            _kept_idx = _sorted[-1]
            _sorted.remove(_kept_idx)
            assert out == {'keep': [_kept_idx], 'delete': _sorted, 'add': None}
            del _sorted, _kept_idx
        elif _trial == 'trial_3':
            # keep last constant column, delete all others
            out = _make_instructions('last', _constant_columns_2)
            _sorted = sorted(list(_constant_columns_2))
            _kept_idx = _sorted[-1]
            _sorted.remove(_kept_idx)
            assert out == {'keep': [_kept_idx], 'delete': _sorted, 'add': None}
            del _sorted, _kept_idx

        # random
        if _trial == 'trial_1':
            # if no constant columns, returns all Nones
            out = _make_instructions('random', _empty_constant_columns)
            assert out == {'keep': None, 'delete': None, 'add': None}
        elif _trial == 'trial_2':
            # keep a random constant column, delete all others
            out = _make_instructions('random', _constant_columns_1)
            # can only validate len of instructions
            assert len(out['keep']) == 1
            assert len(out['delete']) == len(_constant_columns_1) - 1
            assert out['add'] is None
        elif _trial == 'trial_3':
            # keep a random constant column, delete all others
            out = _make_instructions('random', _constant_columns_2)
            # can only validate len of instructions
            assert len(out['keep']) == 1
            assert len(out['delete']) == len(_constant_columns_2) - 1
            assert out['add'] is None

        # none
        if _trial == 'trial_1':
            # if no constant columns, returns all Nones
            out = _make_instructions('none', _empty_constant_columns)
            assert out == {'keep': None, 'delete': None, 'add': None}
        elif _trial == 'trial_2':
            # delete all constant columns
            out = _make_instructions('none', _constant_columns_1)
            assert out == {
                'keep': None,
                'delete': sorted(list(_constant_columns_1)),
                'add': None
            }
        elif _trial == 'trial_3':
            # delete all constant columns
            out = _make_instructions('none', _constant_columns_2)
            assert out == {
                'keep': None,
                'delete': sorted(list(_constant_columns_2)),
                'add': None
            }

        # dict
        if _trial == 'trial_1':
            # if no constant columns, returns all Nones
            out = _make_instructions(_keep_dict, _empty_constant_columns)
            assert out == {'keep': None, 'delete': None, 'add': None}
        elif _trial == 'trial_2':
            # delete all constant columns, append contents of keep dict
            out = _make_instructions(_keep_dict, _constant_columns_1)
            assert out == {
                'keep': None,
                'delete': sorted(list(_constant_columns_1)),
                'add': _keep_dict
            }
        elif _trial == 'trial_3':
            # delete all constant columns, append contents of keep dict
            out = _make_instructions(_keep_dict, _constant_columns_2)
            assert out == {
                'keep': None,
                'delete': sorted(list(_constant_columns_2)),
                'add': _keep_dict
            }












