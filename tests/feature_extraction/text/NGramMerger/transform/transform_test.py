# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.feature_extraction.text._NGramMerger._transform._transform import \
    _transform

import numpy as np

import pytest



class TestTransform:


    # def _transform(
    #     _X: list[list[str]],
    #     _ngrams: Sequence[Sequence[Union[str, re.Pattern]]],
    #     _ngcallable: Union[Callable[[Sequence[str]], str], None],
    #     _sep: Union[str, None]
    # ) -> list[list[str]]:




    @staticmethod
    @pytest.fixture(scope='function')
    def _text():
        return [
            ['NEW', 'YORK', 'CITY', 'NEW', 'YORK'],
            ['FRIED', 'RICE', 'AND', 'MOO', 'GOO', 'GAI', 'PAN'],
            ['BEWARE', 'OF', 'DOG']
        ]


    @staticmethod
    @pytest.fixture(scope='function')
    def _ngrams():
        return [
            ('EGG', 'SALAD'),
            ('NEW', 'YORK'),
            ('NEW', 'YORK', 'CITY'),
            ('FRIED', 'RICE'),
            ('MOO', 'GOO', 'GAI', 'PAN')
        ]


    @pytest.mark.parametrize('_callable, _sep', (
        (None, None),
        (lambda x: "%".join(x), None),
        (None, '@'),
        (None, '&'),
        (None, '__')
    ))
    def test_accuracy(self, _text, _ngrams, _callable, _sep):

        if _callable is not None:
            _exp_sep = '%'
        else:
            _exp_sep = _sep or '_'


        out = _transform(_text, _ngrams, _callable, _sep)

        # new_york_city must trump new_york

        exp = [
            [f'NEW{_exp_sep}YORK{_exp_sep}CITY', f'NEW{_exp_sep}YORK'],
            [f'FRIED{_exp_sep}RICE', 'AND',
             f'MOO{_exp_sep}GOO{_exp_sep}GAI{_exp_sep}PAN'],
            ['BEWARE', 'OF', 'DOG']
        ]

        assert isinstance(out, list)
        for _row_idx in range(len(out)):
            assert isinstance(out[_row_idx], list)
            assert all(map(isinstance, out[_row_idx], (str for _ in out[_row_idx])))
            assert np.array_equal(out[_row_idx], exp[_row_idx])














