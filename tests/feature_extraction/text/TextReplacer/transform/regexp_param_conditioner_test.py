# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

import re

import numpy as np

from pybear.feature_extraction.text._TextReplacer._transform.\
    _regexp_param_conditioner import _regexp_param_conditioner





class TestRegExpParamConditioner:


    @staticmethod
    @pytest.fixture(scope='module')
    def _shape():
        return 10


    @staticmethod
    @pytest.fixture(scope='function')
    def _text(_shape):
        return np.random.choice(list('abcde'), (_shape, ), replace=True)


    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

    @pytest.mark.parametrize('junk_rr',
        (-2.7, -1, 0, 1, 2.7, True, False, 'trash', {'a': 1}, lambda x: x)
    )
    def test_rejects_junk_regexp_replace(self, _text, junk_rr):
        # could be None, tuple, set, list
        with pytest.raises(TypeError):
            _regexp_param_conditioner(None, junk_rr, _text)


    @pytest.mark.parametrize('good_rr',
        (
            None, ('a', ''), ('a', '', 1), {('b', 'B'), (re.compile('c'), 'C')},
            [(re.compile('@'), lambda x: x, 2) for _ in range(10)]
        )
    )
    def test_accepts_good_regexp_replace(self, _text, good_rr):
        # could be None, tuple, set, list
        _regexp_param_conditioner(good_rr, _text)

    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --


    @pytest.mark.parametrize('rr_desc',
        (None, 'tuple_1', 'tuple_2', 'set_1', 'list_1')
    )
    def test_accuracy(self, rr_desc, _text):

        if rr_desc is None:
            rr = None
        elif rr_desc == 'tuple_1':
            rr = ('a', '')
        elif rr_desc == 'tuple_2':
            rr = ('a', '', 1)
        elif rr_desc == 'set_1':
            rr = {('a', ''), ('a', '', 1)}
        elif rr_desc == 'list_1':
            rr = [('b', 'B', 2, re.I) for _ in range(len(_text))]
        else: raise Exception


        out_rr = _regexp_param_conditioner(rr, _text)

        # regexp_replace must always be list(set(tuple(args))))

        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
        assert isinstance(out_rr, list)
        assert len(out_rr) == len(_text)
        assert all(map(isinstance, out_rr, (set for _ in out_rr)))

        if rr_desc is None:
            assert all(map(lambda x: len(x) == 0, out_rr))
        elif rr_desc == 'tuple_1':
            assert all(map(lambda x: x == set((('a', ''),)), out_rr))
        elif rr_desc == 'tuple_2':
            assert all(map(lambda x: x == set((('a', '', 1),)), out_rr))
        elif rr_desc == 'set_1':
            assert all(map(lambda x: x == {('a', ''), ('a', '', 1)}, out_rr))
        elif rr_desc == 'list_1':
            assert all(map(lambda x: x == set((('b', 'B', 2, re.I),)), out_rr))
        else:
            raise Exception





