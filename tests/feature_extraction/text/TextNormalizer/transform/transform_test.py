# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

import numpy as np

from pybear.feature_extraction.text._TextNormalizer._transform._transform \
    import _transform



class TestTransform:


    @staticmethod
    @pytest.fixture(scope='module')
    def _text_1():
        return [
            "A world of dew,",
            "And within every dewdrop",
            "A world of struggle."
        ]


    @staticmethod
    @pytest.fixture(scope='module')
    def _text_1_exp_lower():
        return [
            "a world of dew,",
            "and within every dewdrop",
            "a world of struggle."
        ]


    @staticmethod
    @pytest.fixture(scope='module')
    def _text_1_exp_upper():
        return [
            "A WORLD OF DEW,",
            "AND WITHIN EVERY DEWDROP",
            "A WORLD OF STRUGGLE."
        ]


    @staticmethod
    @pytest.fixture(scope='module')
    def _text_2():
        return [
            ["Even", "more" "so"],
            ["Because", "of", "being", "alone"],
            ["The", "moon", "is", "a", "friend."]
        ]


    @staticmethod
    @pytest.fixture(scope='module')
    def _text_2_exp_lower():
        return [
            ["even", "more" "so"],
            ["because", "of", "being", "alone"],
            ["the", "moon", "is", "a", "friend."]
        ]


    @staticmethod
    @pytest.fixture(scope='module')
    def _text_2_exp_upper():
        return [
            ["EVEN", "MORE" "SO"],
            ["BECAUSE", "OF", "BEING", "ALONE"],
            ["THE", "MOON", "IS", "A", "FRIEND."]
        ]


    def test_accuracy(
        self,
        _text_1, _text_1_exp_lower, _text_1_exp_upper,
        _text_2, _text_2_exp_lower, _text_2_exp_upper
    ):


        # 1D no-op -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
        out = _transform(_text_1, None)
        assert np.array_equal(out, _text_1)
        # END 1D no-op -- -- -- -- -- -- -- -- -- -- -- -- -- -- --


        # 1D lower -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
        out = _transform(_text_1, False)
        assert np.array_equal(out, _text_1_exp_lower)
        # END 1D lower -- -- -- -- -- -- -- -- -- -- -- -- -- -- --


        # 1D upper -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
        out = _transform(_text_1, True)
        assert np.array_equal(out, _text_1_exp_upper)
        # END 1D upper -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

        # _transform can only take 1D. 2D is handled recursively in
        # the main transform() method.

        # 2D no-op -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
        with pytest.raises(ValueError):
            _transform(_text_2, None)
        # END 2D no-op -- -- -- -- -- -- -- -- -- -- -- -- -- -- --


        # 2D lower -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
        with pytest.raises(ValueError):
            _transform(_text_2, False)
        # END 2D lower -- -- -- -- -- -- -- -- -- -- -- -- -- -- --


        # 2D upper -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
        with pytest.raises(ValueError):
            _transform(_text_2, True)
        # END 2D upper -- -- -- -- -- -- -- -- -- -- -- -- -- -- --





