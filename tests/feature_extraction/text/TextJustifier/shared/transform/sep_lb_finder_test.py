# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.feature_extraction.text._TextJustifier._shared._transform. \
    _sep_lb_finder import _sep_lb_finder

import pytest

import re

import numpy as np



class TestSepLbFinder:


    # def _sep_lb_finder(
    #     _X: XWipContainer,
    #     _join_2D: str,
    #     _sep: Union[re.Pattern[str], tuple[re.Pattern[str], ...]],
    #     _line_break: Union[None, re.Pattern[str], tuple[re.Pattern[str], ...]]
    # ) -> list[bool]:


    @staticmethod
    @pytest.fixture(scope='function')
    def _text():

        # _sep_lb_finder will only ever see 1D data

        return [
            "THESE are the times that try men’s souls: The summer ",
            "soldier and the sunshine patriot will, in this crisis, shrink",
            "from the service of his country; but he that stands it now,",
            "deserves the love and thanks of man and woman."
        ]


    @pytest.mark.parametrize('_join_2D', (' ,', ' ', '.'))
    @pytest.mark.parametrize('_sep',
        (re.compile(','), re.compile(' '), (re.compile(','), re.compile(' ')))
    )
    @pytest.mark.parametrize('_lb',
        (re.compile('\.'), re.compile(';'), (re.compile('\.'), re.compile(';')), None)
    )
    def test_accuracy(self, _text, _join_2D, _sep, _lb):

        # line 1 ends with " "
        # line 2 ends with "k"
        # line 3 ends with ","
        # line 4 ends with "."

        _ref = [False] * len(_text)

        # if _join_2D == ' ,', then no match against _sep or _lb
        if _join_2D == ' ,':
            pass
        else:
            # line 1
            if _join_2D == ' ' and (_sep == re.compile(' ')
                    or (isinstance(_sep, tuple) and re.compile(' ') in _sep)):
                _ref[0] = True

            # line 2
            # can never become True

            # line 3
            # join_2D never == ',', so can never be True

            # line 4
            if _join_2D == '.' and (_lb == re.compile('\.')
                    or (isinstance(_lb, tuple) and re.compile('\.') in _lb)):
                _ref[3] = True



        out = _sep_lb_finder(
            _text,
            _join_2D,
            _sep,
            _lb
        )

        assert isinstance(out, list)
        assert all(map(isinstance, out, (bool for _ in out)))
        assert np.array_equal(out, _ref)






