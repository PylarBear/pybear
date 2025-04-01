# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.feature_extraction.text._TextRemover._transform._param_conditioner. \
    _param_conditioner import _param_conditioner

import pytest

import re

import numpy as np



class TestParamConditioner:

    # def _param_conditioner(
    #     _remove: RemoveType,
    #     _case_sensitive: CaseSensitiveType,
    #     _flags: FlagsType,
    #     _n_rows: numbers.Integral
    # ) -> Union[
    #         None, re.Pattern[str], tuple[re.Pattern, ...],
    #         list[Union[None, re.Pattern[str], tuple[re.Pattern, ...]]]
    #     ]:

    # no validation

    # we know the submodules that comprise _param_conditioner work because
    # of their own tests. just test that this gives the expected output.

    @staticmethod
    @pytest.fixture(scope='function')
    def _remove_as_list():
        return [
            None,
            'remove_me',
            re.compile('take_me_out', re.I),
            ('im_a_gonner', re.compile('go_bye_bye'))
        ]


    @staticmethod
    @pytest.fixture(scope='function')
    def _n_rows(_remove_as_list):
        return len(_remove_as_list)


    @pytest.mark.parametrize('_remove_format',
        ('None', 'str', 'compile', 'tuple', 'list')
    )
    @pytest.mark.parametrize('_cs', (True, False, [None, True, False, None]))
    @pytest.mark.parametrize('_flags', (None, re.I, [re.I, None, re.M, None]))
    def test_accuracy(
        self, _remove_as_list, _n_rows, _remove_format, _cs, _flags
    ):

        if _remove_format == 'None':
            _remove = None
            _exp_out = None
        elif _remove_format == 'str':
            _remove = 'take_me_out'
            _exp_out = re.compile('take_me_out')
        elif _remove_format == 'compile':
            _remove = re.compile('go_away', re.I)
            _exp_out = re.compile('go_away', re.I)
        elif _remove_format == 'tuple':
            _remove = ('no_mas', re.compile('game_over'))
            _exp_out = (re.compile('no_mas'), re.compile('game_over'))
        elif _remove_format == 'list':
            _remove = _remove_as_list
            _exp_out = [
                None,
                re.compile('remove_me'),
                re.compile('take_me_out', re.I),
                (re.compile('im_a_gonner'), re.compile('go_bye_bye'))
            ]
        else:
            raise Exception

        out = _param_conditioner(
            _remove,
            _cs,
            _flags,
            _n_rows
        )

        # forget about the flags, we know flag_maker works right
        if _remove is None:
            assert out is None
        elif isinstance(_remove, str):
            if isinstance(out, re.Pattern):
                assert out.pattern == _exp_out.pattern
            else:  # must be list because of different flags
                for i in out:
                    assert i.pattern == _exp_out.pattern
        elif isinstance(_remove, re.Pattern):
            if isinstance(out, re.Pattern):
                assert out.pattern == _exp_out.pattern
            else:  # must be list because of different flags
                for i in out:
                    assert i.pattern == _exp_out.pattern
        elif isinstance(_remove, tuple):
            if isinstance(out, tuple):
                _out_patterns = [i.pattern for i in out]
                _exp_patterns = [j.pattern for j in _exp_out]
                assert np.array_equal(
                    sorted(_out_patterns),
                    sorted(_exp_patterns)
                )
            else:  # must be list because of different flags
                for _tuple in out:
                    _out_patterns = [i.pattern for i in _tuple]
                    _exp_patterns = [j.pattern for j in _exp_out]
                    assert np.array_equal(
                        sorted(_out_patterns),
                        sorted(_exp_patterns)
                    )
        elif isinstance(_remove, list):
            for _idx, _thing in enumerate(out):
                if _thing is None:
                    assert _exp_out[_idx] is None
                elif isinstance(_thing, re.Pattern):
                    assert _thing.pattern == _exp_out[_idx].pattern
                elif isinstance(_thing, tuple):
                    _out_patterns = [i.pattern for i in _thing]
                    _exp_patterns = [j.pattern for j in _exp_out[_idx]]
                    assert np.array_equal(
                        sorted(_out_patterns),
                        sorted(_exp_patterns)
                    )
                else:
                    raise Exception
        else:
            raise Exception





