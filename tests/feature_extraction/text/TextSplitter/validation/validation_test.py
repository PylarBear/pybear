# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

import re

from pybear.feature_extraction.text._TextSplitter._validation._validation \
    import _validation



class TestValidation:


    # the brunt of validation is handled in the individual module


    # check check certain access is blocked

    @pytest.mark.parametrize('str_sep', (' ', {' ', ', '}, [' ', ', '], None))
    @pytest.mark.parametrize('str_maxsplit', (-1, [1,2], None))
    @pytest.mark.parametrize('regexp_sep',
         ('\W', re.compile('\W'), [re.compile('\d'), '[a-m]*'], None)
    )
    @pytest.mark.parametrize('regexp_maxsplit', (-1, [1,2], None))
    @pytest.mark.parametrize('regexp_flags', (-1, [1, 2], None))
    def test_accuracy(
        self, str_sep, str_maxsplit, regexp_sep, regexp_maxsplit, regexp_flags
    ):

        _will_raise = False

        if any((str_sep, str_maxsplit)) \
                and any((regexp_sep, regexp_maxsplit, regexp_flags)):
            _will_raise = True

        elif any((regexp_maxsplit, regexp_flags)) and not regexp_sep:

            _will_raise = True


        if _will_raise:
            with pytest.raises(ValueError):
                _validation(
                    list('ab'),
                    str_sep,
                    str_maxsplit,
                    regexp_sep,
                    regexp_maxsplit,
                    regexp_flags
                )
        else:
            out = _validation(
                list('ab'),
                str_sep,
                str_maxsplit,
                regexp_sep,
                regexp_maxsplit,
                regexp_flags
            )

            assert out is None


    # check catches bad 'False' distribution

    str_types = ('type_1_str', 'type_2_str', 'type_3_str', 'type_4_str')
    int_types = ('type_1_int', 'type_2_int', 'type_3_int', 'type_4_int')


    @staticmethod
    @pytest.fixture(scope='module')
    def DICT():
        return {
            'type_1_str': [False, ',', ','],
            'type_2_str': [',', False, ','],
            'type_3_str': [False, ',', ','],
            'type_4_str': ',',
            'type_1_int': [False, 0, 0],
            'type_2_int': [0, False, 0],
            'type_3_int': [False, 0, 0],
            'type_4_int': 0
        }


    @pytest.mark.parametrize('str_sep', str_types)
    @pytest.mark.parametrize('str_maxsplit', int_types)
    def test_false_distribution_str(self, DICT, str_sep, str_maxsplit):

        # type_1 vs type_2 will raise
        # type_2 vs type_3 will raise


        _will_raise = False

        if str_sep in ['type_1_str', 'type_3_str'] \
                and str_maxsplit == 'type_2_int':
            _will_raise = True

        if str_maxsplit in ['type_1_int', 'type_3_int'] \
                and str_sep == 'type_2_str':
            _will_raise = True


        if _will_raise:
            with pytest.raises(ValueError):
                _validation(
                    list('abc'),
                    _str_sep=DICT[str_sep],
                    _str_maxsplit=DICT[str_maxsplit],
                    _regexp_sep=None,
                    _regexp_maxsplit=None,
                    _regexp_flags=None
                )
        else:
            out = _validation(
                list('abc'),
                _str_sep=DICT[str_sep],
                _str_maxsplit=DICT[str_maxsplit],
                _regexp_sep=None,
                _regexp_maxsplit=None,
                _regexp_flags=None
            )

            assert out is None



    @pytest.mark.parametrize('regexp_sep', str_types)
    @pytest.mark.parametrize('regexp_maxsplit', int_types)
    @pytest.mark.parametrize('regexp_flags', int_types)
    def test_false_distribution_regexp(
        self, DICT, regexp_sep, regexp_maxsplit, regexp_flags
    ):


        _will_raise = False

        _a = 'type_1' in regexp_sep or 'type_3' in regexp_sep
        _b = 'type_1' in regexp_maxsplit or 'type_3' in regexp_maxsplit
        _c = 'type_1' in regexp_flags or 'type_3' in regexp_flags
        _d = 'type_2' in regexp_sep
        _e = 'type_2' in regexp_maxsplit
        _f = 'type_2' in regexp_flags

        if any((_a, _b, _c)) and any((_d, _e, _f)):
            _will_raise = True

        elif any((regexp_maxsplit, regexp_flags)) and not regexp_sep:

            _will_raise = True


        if _will_raise:
            with pytest.raises(ValueError):
                _validation(
                    list('abc'),
                    _str_sep=None,
                    _str_maxsplit=None,
                    _regexp_sep=DICT[regexp_sep],
                    _regexp_maxsplit=DICT[regexp_maxsplit],
                    _regexp_flags=DICT[regexp_flags]
                )
        else:
            out = _validation(
                list('abc'),
                _str_sep=None,
                _str_maxsplit=None,
                _regexp_sep=DICT[regexp_sep],
                _regexp_maxsplit=DICT[regexp_maxsplit],
                _regexp_flags=DICT[regexp_flags]
            )

            assert out is None





