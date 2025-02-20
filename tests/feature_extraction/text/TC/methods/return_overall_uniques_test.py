# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

import numpy as np

from pybear.feature_extraction.text._TC._methods._return_overall_uniques__NOT_USED \
    import _return_overall_uniques

from pybear.feature_extraction.text._TextStatistics.TextStatistics import \
    TextStatistics as TS




class TestReturnOverallUniques:


    @staticmethod
    @pytest.fixture(scope='module')
    def _text():
        __ = [
            "you have brains in your head",
            "you have feet in your shoes",
            "you can steer yourself",
            "any direction you choose",
            "you're on your own",
            "and you know what you know",
            "and you are the guy who'll decide where to go"
        ]

        return list(map(str.split, __))


    @staticmethod
    @pytest.fixture(scope='module')
    def _uniques(_text):

        ts = TS(store_uniques=True)

        for _list in _text:
            ts.partial_fit(_list)

        return ts.string_frequency_



    # validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    @pytest.mark.parametrize('junk_X',
        (-2.7, -1, 0, 1, 2.7, True, None, 'trash', [0,1], (1,), {1,2},
         {'A':1}, lambda x: x, ['a', 'b'])
    )
    def test_rejects_junk_X(self, junk_X):

        with pytest.raises(TypeError):
            _return_overall_uniques(
                junk_X,
                _return_counts=False
            )


    @pytest.mark.parametrize('junk_rc',
        (-2.7, -1, 0, 1, 2.7, None, 'trash', [0,1], (1,), {1,2}, {'A':1},
         lambda x: x)
    )
    def test_rejects_junk_rc(self, junk_rc):

        with pytest.raises(TypeError):
            _return_overall_uniques(
                list('abcde'),
                _return_counts=junk_rc
            )

    # END validation ** * ** * ** * ** * ** * ** * ** * ** * ** * **



    @pytest.mark.parametrize('return_counts', (True, False))
    def test_accuracy(self, _text, _uniques, return_counts):

        out = _return_overall_uniques(_text, return_counts)


        if return_counts is False:
            assert isinstance(out, np.ndarray)
            assert np.array_equal(
                out,
                np.sort(list(_uniques.keys()))
            )

        elif return_counts is True:
            assert isinstance(out, tuple)

            _argsort = np.argsort(np.array(list(_uniques.keys())))

            assert isinstance(out[0], np.ndarray)
            assert np.array_equal(
                out[0],
                np.array(list(_uniques.keys()))[_argsort]
            )

            assert isinstance(out[1], np.ndarray)
            assert np.array_equal(
                out[1],
                np.array(list(_uniques.values()))[_argsort]
            )












