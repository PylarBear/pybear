# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

import re

import numpy as np

from pybear.feature_extraction.text._TextReplacer._transform._str_param_conditioner \
    import _str_param_conditioner



class TestStrParamConditioner:


    @staticmethod
    @pytest.fixture(scope='module')
    def _shape():
        return 10


    @staticmethod
    @pytest.fixture(scope='function')
    def _text(_shape):
        return np.random.choice(list('abcde'), (_shape, ), replace=True)


    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

    @pytest.mark.parametrize('junk_sr',
        (-2.7, -1, 0, 1, 2.7, True, False, 'trash', {'a': 1}, lambda x: x)
    )
    def test_rejects_junk_str_replace(self, _text, junk_sr):
        # could be None, tuple, set, list
        with pytest.raises(TypeError):
            _str_param_conditioner(junk_sr, None, _text)


    @pytest.mark.parametrize('good_sr',
        (
            None, ('a', ''), ('a', '', 1), {('b', 'B'), ('c', 'C')},
            [('@', '') for _ in range(10)]
        )
    )
    def test_accepts_good_str_replace(self, _text, good_sr):
        # could be None, tuple, set, list
        _str_param_conditioner(good_sr, _text)

    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --


    @pytest.mark.parametrize('sr_desc',
        (None, 'tuple_1', 'tuple_2', 'set_1','list_1')
    )
    @pytest.mark.parametrize('rr_desc',
        (None, 'tuple_1', 'tuple_2', 'set_1', 'list_1')
    )
    def test_accuracy(self, sr_desc, rr_desc, _text):

        if sr_desc is None:
            sr = None
        elif sr_desc == 'tuple_1':
            sr = ('a', '')
        elif sr_desc == 'tuple_2':
            sr = ('a', '', 1)
        elif sr_desc == 'set_1':
            sr = {('a', ''), ('a', '', 1)}
        elif sr_desc == 'list_1':
            sr = [('b', 'B', 2) for _ in range(len(_text))]
        else: raise Exception


        out_sr = _str_param_conditioner(sr, _text)

        # str_replace must always be list(set(tuple(args))))

        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
        assert isinstance(out_sr, list)
        assert len(out_sr) == len(_text)
        assert all(map(isinstance, out_sr, (set for _ in out_sr)))

        if sr_desc is None:
            assert all(map(lambda x: len(x) == 0, out_sr))
        elif sr_desc == 'tuple_1':
            assert all(map(lambda x: x == set((('a', ''),)), out_sr))
        elif sr_desc == 'tuple_2':
            assert all(map(lambda x: x == set((('a', '', 1),)), out_sr))
        elif sr_desc == 'set_1':
            assert all(map(lambda x: x == {('a', ''), ('a', '', 1)}, out_sr))
        elif sr_desc == 'list_1':
            assert all(map(lambda x: x == set((('b', 'B', 2),)), out_sr))
        else:
            raise Exception






