# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

import re

import numpy as np

from pybear.feature_extraction.text._TextLookup._shared._validation._validation \
    import _validation



class TestValidation:


    # the brunt of the work is handled by the individual validation modules,
    # which are tested separately. just make sure validation passes all good
    # and correctly handles the interdependency of parameters and the
    # specially handled words.


    @pytest.mark.parametrize('_update_lexicon', (True,))
    @pytest.mark.parametrize('_skip_numbers', (False,))
    @pytest.mark.parametrize('_auto_split', (True,))
    @pytest.mark.parametrize('_auto_add_to_lexicon', (False,))
    @pytest.mark.parametrize('_auto_delete', (True,))
    @pytest.mark.parametrize('_DELETE_ALWAYS', ('list1', 'list2', 'list3', 'none'))
    @pytest.mark.parametrize('_REPLACE_ALWAYS', ('dict1', 'dict2', 'dict3', 'none'))
    @pytest.mark.parametrize('_SKIP_ALWAYS', ('list1', 'list2', 'list3', 'none'))
    @pytest.mark.parametrize('_SPLIT_ALWAYS', ('dict1', 'dict2', 'dict3', 'none'))
    @pytest.mark.parametrize('_remove_empty_rows', (False,))
    @pytest.mark.parametrize('_verbose', (True,))
    def test_accuracy(
        self, _update_lexicon, _skip_numbers, _auto_split, _auto_add_to_lexicon,
        _auto_delete, _DELETE_ALWAYS, _REPLACE_ALWAYS, _SKIP_ALWAYS, _SPLIT_ALWAYS,
        _remove_empty_rows, _verbose
    ):

        # set parameters ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
        if _DELETE_ALWAYS == 'list1':
            _DELETE_ALWAYS = list('abc')
        elif _DELETE_ALWAYS == 'list2':
            _DELETE_ALWAYS = list('mno')
        elif _DELETE_ALWAYS == 'list3':
            _DELETE_ALWAYS = ['m', 'n', re.compile('^o+$', re.I)]
        elif _DELETE_ALWAYS == 'none':
            _DELETE_ALWAYS = None
        else:
            raise Exception

        if _REPLACE_ALWAYS == 'dict1':
            _REPLACE_ALWAYS = dict((zip(list('abc'), list('123'))))
        elif _REPLACE_ALWAYS == 'dict2':
            _REPLACE_ALWAYS = dict((zip(list('pqr'), list('123'))))
        elif _REPLACE_ALWAYS == 'dict3':
            _REPLACE_ALWAYS = dict((zip(
                ['p', 'q', re.compile('^r+$', re.I)],
                list('123')
            )))
        elif _REPLACE_ALWAYS == 'none':
            _REPLACE_ALWAYS = None
        else:
            raise Exception

        if _SKIP_ALWAYS == 'list1':
            _SKIP_ALWAYS = list('abc')
        elif _SKIP_ALWAYS == 'list2':
            _SKIP_ALWAYS = list('stu')
        elif _SKIP_ALWAYS == 'list3':
            _SKIP_ALWAYS = ['s', 't', re.compile('^u+$', re.I)]
        elif _SKIP_ALWAYS == 'none':
            _SKIP_ALWAYS = None
        else:
            raise Exception

        if _SPLIT_ALWAYS == 'dict1':
            _SPLIT_ALWAYS = dict((zip(list('abc'), ([], [], []))))
        elif _SPLIT_ALWAYS == 'dict2':
            _SPLIT_ALWAYS =  dict((zip(list('vwx'), ([], [], []))))
        elif _SPLIT_ALWAYS == 'dict3':
            _SPLIT_ALWAYS = dict((zip(
                ['v', 'w', re.compile('^x+$', re.I)],
                ([], [], [])
            )))
        elif _SPLIT_ALWAYS == 'none':
            _SPLIT_ALWAYS = None
        else:
            raise Exception
        # END set parameters ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *



        _X = np.random.choice(list('abcde'), (5, 3), replace=True)

        _raise_for_parameter_conflict = 0
        if _auto_add_to_lexicon and not _update_lexicon:
            _raise_for_parameter_conflict += 1
        if _auto_delete and _update_lexicon:
            _raise_for_parameter_conflict += 1


        _raise_for_duplicate_in_special = False
        _equal_abc = 0
        for i in (_DELETE_ALWAYS, _REPLACE_ALWAYS, _SKIP_ALWAYS, _SPLIT_ALWAYS):
            if i is None:
                continue
            # the 4 test 'ALWAYS' objects are rigged so that when they hold, or
            # are keyed by, 'a', 'b', and 'c', then they could be duplicate
            if list(i) == list('abc'):
                _equal_abc += 1
        # if 2+ 'ALWAYS' objects have 'a' 'b' 'c', then there is a conflict
        if _equal_abc >= 2:
            _raise_for_duplicate_in_special = True


        if _raise_for_parameter_conflict:
            with pytest.raises(ValueError):
                _validation(
                    _X,
                    _update_lexicon,
                    _skip_numbers,
                    _auto_split,
                    _auto_add_to_lexicon,
                    _auto_delete,
                    _DELETE_ALWAYS,
                    _REPLACE_ALWAYS,
                    _SKIP_ALWAYS,
                    _SPLIT_ALWAYS,
                    _remove_empty_rows,
                    _verbose
                )
        elif _raise_for_duplicate_in_special:
            with pytest.raises(ValueError):
                _validation(
                    _X,
                    _update_lexicon,
                    _skip_numbers,
                    _auto_split,
                    _auto_add_to_lexicon,
                    _auto_delete,
                    _DELETE_ALWAYS,
                    _REPLACE_ALWAYS,
                    _SKIP_ALWAYS,
                    _SPLIT_ALWAYS,
                    _remove_empty_rows,
                    _verbose
                )
        else:
            out = _validation(
                _X,
                _update_lexicon,
                _skip_numbers,
                _auto_split,
                _auto_add_to_lexicon,
                _auto_delete,
                _DELETE_ALWAYS,
                _REPLACE_ALWAYS,
                _SKIP_ALWAYS,
                _SPLIT_ALWAYS,
                _remove_empty_rows,
                _verbose
            )

            assert out is None







