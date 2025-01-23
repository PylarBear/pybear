# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.preprocessing.MinCountTransformer._validation. \
    _count_threshold import _val_count_threshold

import re
import numbers
import numpy as np

import pytest



class TestValCountThreshold:


    @staticmethod
    @pytest.fixture(scope='module')
    def err_quip_1():
        return "is passed as a single integer it must be"


    @staticmethod
    @pytest.fixture(scope='module')
    def err_quip_2():
        return "the length of the iterable also must "


    @pytest.mark.parametrize('_junk_count_threshold',
        (None, 'junk', {'a':1}, lambda x: x)
)
    def test_junk_count_threshold(
        self, _junk_count_threshold, err_quip_1, err_quip_2
    ):

        # valid is single number or could be iterable of numbers

        with pytest.raises(TypeError) as exc:
            _val_count_threshold(
                _count_threshold=_junk_count_threshold,
                _n_features_in=5
            )

        # these are iterable so would enter into the iterable handling and
        # return errors only for iterables
        if isinstance(_junk_count_threshold, (str, dict)):
            assert re.escape(err_quip_1) not in re.escape(str(exc))
            assert re.escape(err_quip_2) in re.escape(str(exc))
        else:  # the rest will bounce of both and return both error messages
            assert re.escape(err_quip_1) in re.escape(str(exc))
            assert re.escape(err_quip_2) in re.escape(str(exc))


    @pytest.mark.parametrize('_bad_count_threshold',
        (-2.7, -1, 0, 1, 2.7, True, False)
    )
    def test_bad_count_threshold_as_single_value(
        self, _bad_count_threshold, err_quip_1, err_quip_2
    ):

        # must be integer >= 2

        if isinstance(_bad_count_threshold, bool):

            with pytest.raises(TypeError) as exc:
                _val_count_threshold(
                    _bad_count_threshold,
                    _n_features_in=5
                )
        else:
            with pytest.raises(ValueError) as exc:
                _val_count_threshold(
                    _bad_count_threshold,
                    _n_features_in=5
                )

        assert re.escape(err_quip_1) in re.escape(str(exc))
        assert re.escape(err_quip_2) not in re.escape(str(exc))


    @pytest.mark.parametrize('_bad_count_threshold',
        (
            [-1, 0, 1],      # 2 numbers below 1, no number >= 2
            [2.7, 2.8, 2.9],  # floats
            list('abc'),      # strings
            np.random.randint(0, 10, (3, 3)),     # bad shape
            [2, 3, 4, 5, 6]      # bad len
        )
    )
    def test_bad_count_threshold_as_iterable(
        self, _bad_count_threshold, err_quip_1, err_quip_2
    ):

        # must be 1D, only integers >= 1 with at least 1 >= 2

        with pytest.raises(ValueError) as exc:
            _val_count_threshold(
                _bad_count_threshold,
                _n_features_in=3
            )

        assert re.escape(err_quip_1) not in re.escape(str(exc))
        assert re.escape(err_quip_2) in re.escape(str(exc))


    @pytest.mark.parametrize('_count_threshold',
        (2, [1, 1, 2], [10, 20, 30])
    )
    def test_accepts_good_count_threshold(self, _count_threshold):
        _val_count_threshold(
            _count_threshold,
            _n_features_in=3
        )











