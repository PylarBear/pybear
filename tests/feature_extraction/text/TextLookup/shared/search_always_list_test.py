# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

import re

from pybear.feature_extraction.text._TextLookup._shared._search_always_list \
    import _search_always_list



class TestSearchSpecialHandleList:


    @pytest.mark.parametrize('_name',
        ('SKIP_ALWAYS', 'DELETE_ALWAYS', 'REPLACE_ALWAYS', 'SPLIT_ALWAYS'
         'fries', 'nuggets', 'soda', True, False, None, lambda x: x)
    )
    @pytest.mark.parametrize('_LIST',
        ({'a': 1}, 'foobar', ['baz', 'quux'], True, None, lambda x: x)
    )
    @pytest.mark.parametrize('_word',
        (0, 1, 2.7, True, None, [0,1], 'fish', 'salad')
    )
    def test_rejects_junk(self, _name, _LIST, _word):

        _will_raise = 0
        if _name not in [
            'SKIP_ALWAYS', 'DELETE_ALWAYS', 'REPLACE_ALWAYS', 'SPLIT_ALWAYS'
        ]:
            _will_raise += 1
        if not isinstance(_LIST, list):
            _will_raise += 1
        if not isinstance(_word, str):
            _will_raise += 1

        if _will_raise:
            with pytest.raises(Exception):
                _search_always_list(_name, _LIST, _word)
        else:
            assert _search_always_list(_name, _LIST, _word) is False


    @pytest.mark.parametrize('_name',
        ('SKIP_ALWAYS', 'DELETE_ALWAYS', 'REPLACE_ALWAYS', 'SPLIT_ALWAYS')
    )
    @pytest.mark.parametrize('_LIST,_word,_isin',
        (
            ([], 'subterfuge', False),
            (['eggs', 'chicken', 'bacon'], 'bacon', True),
            (['eggs', 'chicken', 'bacon'], 'BACON', False),
            ([re.compile('ENGINEER.+'), 'FOO'], 'ENGINEERING', True),
            ([re.compile('engineer.+'), 'FOO'], 'ENGINEERING', False)
        )
    )
    def test_accuracy(self, _name, _LIST, _word, _isin):

        out = _search_always_list(_name, _LIST, _word)

        assert out is _isin


    @pytest.mark.parametrize('_name',
        ('SKIP_ALWAYS', 'DELETE_ALWAYS', 'REPLACE_ALWAYS', 'SPLIT_ALWAYS')
    )
    def test_catches_multiple_matches(self, _name):

        # conflict between 2+ re.compiles
        with pytest.raises(ValueError):
            _search_always_list(
                _name,
                [re.compile('^eg+$'), re.compile('^egg$')],
                'egg'
            )

        # conflict between literal str and a re.compile
        with pytest.raises(ValueError):
            _search_always_list(
                _name,
                [re.compile(f'^eg+$'), 'egg'],
                'egg'
            )





