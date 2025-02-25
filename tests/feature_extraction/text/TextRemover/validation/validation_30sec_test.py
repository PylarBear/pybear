# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

import re

import numpy as np

from pybear.feature_extraction.text._TextRemover._validation._validation import \
    _validation



class TestValidation:

    # the brunt of testing validation is handled by the individual validation
    # modules' tests

    # just make sure that it accepts all good parameters, and that the
    # conditionals for passing parameters are enforced.

    # 1) cannot pass str & re kwargs simultaneously
    # 2) cannot pass 'regexp_flags' alone
    # 3) cannot leave all defaults


    @staticmethod
    @pytest.fixture(scope='module')
    def _text_dim_1():
        return [
            'Despair thy charm, ',
            'And let the angel whom thou still hast served ',
            'Tell thee Macduff was ',
            "from his mother’s womb",
            "Untimely ripped."
        ]


    @staticmethod
    @pytest.fixture(scope='module')
    def _text_dim_2():
        return [
            ['Despair', 'thy' 'charm, '],
            ['And', 'let', 'the', 'angel', 'whom', 'thou', 'still', 'hast' 'served '],
            ['Tell', 'thee', 'Macduff', 'was '],
            ['from', 'his', "mother’s", 'womb'],
            ['Untimely', 'ripped.']
        ]


    @staticmethod
    @pytest.fixture(scope='function')
    def str_remove_seq_1():
        return [{' ', ',', '.'}, '', '\n', '\s', False]

    @staticmethod
    @pytest.fixture(scope='function')
    def re_remove_seq_1():
        return [False, re.compile('[n-z]'), '[n-z]', '[n-z]', '[n-z]']

    @staticmethod
    @pytest.fixture(scope='function')
    def re_remove_seq_2():
        return [re.compile('[n-z]'), False, '[n-z]', '[n-z]', '[n-z]']

    @staticmethod
    @pytest.fixture(scope='function')
    def re_flags_seq_1():
        return [False, re.I, re.X, re.I | re.X, None]

    @staticmethod
    @pytest.fixture(scope='function')
    def re_flags_seq_2():
        return [re.I, False, re.X, None, re.I | re.X]


    @pytest.mark.parametrize('dim', (1, 2))
    @pytest.mark.parametrize('X_container', (list, set, tuple, np.ndarray))
    @pytest.mark.parametrize('str_remove_container', (list, tuple))
    @pytest.mark.parametrize('str_remove',
        (None, ' ', {' ', '\n', '\s'}, 'str_remove_seq_1')
    )
    @pytest.mark.parametrize('re_remove_container', (list, tuple))
    @pytest.mark.parametrize('re_remove',
        (None, '[a-m]', re.compile('[A-M]'), 're_remove_seq_1', 're_remove_seq_2')
    )
    @pytest.mark.parametrize('re_flags_container', (list, tuple))
    @pytest.mark.parametrize('re_flags',
        (None, re.I | re.X, 're_flags_seq_1', 're_flags_seq_2')
    )
    def test_accuracy(
        self,
        dim, X_container, str_remove, re_remove, re_flags,
        _text_dim_1, _text_dim_2,
        str_remove_container, re_remove_container, re_flags_container,
        str_remove_seq_1, re_remove_seq_1, re_remove_seq_2, re_flags_seq_1,
        re_flags_seq_2
    ):

        if dim == 2 and X_container is set:
            pytest.skip(reason=f'impossible condition')

        # END skip impossible -- -- -- -- -- -- -- -- -- -- -- -- -- --


        _type_error = False
        _value_error = False

        if dim == 1:
            if X_container is np.ndarray:
                _X = np.array(_text_dim_1)
            else:
                _X = X_container(_text_dim_1)
            assert isinstance(_X, X_container)
        elif dim == 2:
            if X_container is np.ndarray:
                _X = np.fromiter(map(lambda x: np.array(x), _text_dim_2), dtype=object)
            else:
                _X = X_container(map(X_container, _text_dim_2))
            assert isinstance(_X, X_container)
            assert all(map(isinstance, _X, (X_container for _ in _X)))
        else:
            raise Exception

        if str_remove == 'str_remove_seq_1':
            str_remove = str_remove_container(str_remove_seq_1)
            assert isinstance(str_remove, str_remove_container)

        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
        if re_remove == 're_remove_seq_1' \
                and re_flags == 're_flags_seq_2':
            _value_error = True

        if re_remove == 're_remove_seq_2' \
                and re_flags == 're_flags_seq_1':
            _value_error = True
        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

        if re_remove == 're_remove_seq_1':
            re_remove = re_remove_container(re_remove_seq_1)
            assert isinstance(re_remove, re_remove_container)
        elif re_remove == 're_remove_seq_2':
            re_remove = re_remove_container(re_remove_seq_2)
            assert isinstance(re_remove, re_remove_container)


        if re_flags == 're_flags_seq_1':
            re_flags = re_flags_container(re_flags_seq_1)
            assert isinstance(re_flags, re_flags_container)
        elif re_flags == 're_flags_seq_2':
            re_flags = re_flags_container(re_flags_seq_2)
            assert isinstance(re_flags, re_flags_container)

        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

        if str_remove and any((re_remove, re_flags)):
            _value_error = True

        if not any((str_remove, re_remove, re_flags)):
            _value_error = True

        if re_flags and not re_remove:
            _value_error = True

        if isinstance(str_remove, tuple):
            _type_error = True

        if isinstance(re_remove, (set, tuple)):
            _type_error = True

        if isinstance(re_flags, (set, tuple)):
            _type_error = True

        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

        if _type_error or _value_error:
            with pytest.raises((TypeError, ValueError)):
                _validation(
                    _X,
                    _str_remove=str_remove,
                    _regexp_remove=re_remove,
                    _regexp_flags=re_flags
                )
        else:
            out = _validation(
                _X,
                _str_remove=str_remove,
                _regexp_remove=re_remove,
                _regexp_flags=re_flags
            )


            assert out is None





