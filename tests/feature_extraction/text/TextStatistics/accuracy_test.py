# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest
from copy import deepcopy

import numpy as np

from pybear.feature_extraction.text._TextStatistics.TextStatistics import \
    TextStatistics as TS

from ._read_green_eggs_and_ham import _read_green_eggs_and_ham



class TestTextStatistics:


    @staticmethod
    @pytest.fixture(scope='module')
    def STRINGS():

        return _read_green_eggs_and_ham()


    def test_multiple_partial_fit_accuracy(self, STRINGS):

        # fit data with partial_fit
        # store the attributes
        # run another partial_fit with the same data
        # store the attributes

        # size_ should double
        # uniques_ should be the same
        # overall_statistics_
        # - 'size' should double
        # - 'uniques_count' should stay the same
        # - 'max_length' should stay the same
        # - 'min_length' should stay the same
        # - 'average_length' should stay the same
        # - 'std_length' should stay the same
        # string_frequency_ should double
        # startswith_frequency_ should double
        # character_frequency_ should double



        TestCls = TS()

        TestCls.partial_fit(STRINGS)

        fst_size = deepcopy(TestCls.size_)
        fst_uniques = deepcopy(TestCls.uniques_)
        fst_overall_statistics = deepcopy(TestCls.overall_statistics_)
        fst_string_frequency = deepcopy(TestCls.string_frequency_)
        fst_startswith_frequency = deepcopy(TestCls.startswith_frequency_)
        fst_character_frequency = deepcopy(TestCls.character_frequency_)

        TestCls.partial_fit(STRINGS)

        scd_size = TestCls.size_
        scd_uniques = TestCls.uniques_
        scd_overall_statistics = TestCls.overall_statistics_
        scd_string_frequency = TestCls.string_frequency_
        scd_startswith_frequency = TestCls.startswith_frequency_
        scd_character_frequency = TestCls.character_frequency_


        # size_ should double
        assert scd_size == 2 * fst_size

        # uniques_ should be the same
        assert np.array_equal(scd_uniques, fst_uniques)

        # overall_statistics_
        # - 'size' should double
        assert scd_overall_statistics['size'] == 2 * fst_overall_statistics['size']

        # - 'uniques_count' should stay the same
        assert scd_overall_statistics['uniques_count'] == \
               fst_overall_statistics['uniques_count']

        # - 'max_length' should stay the same
        assert scd_overall_statistics['max_length'] == \
               fst_overall_statistics['max_length']

        # - 'min_length' should stay the same
        assert scd_overall_statistics['min_length'] == \
               fst_overall_statistics['min_length']

        # - 'average_length' should stay the same
        assert scd_overall_statistics['average_length'] == \
               fst_overall_statistics['average_length']

        # - 'std_length' should stay the same
        assert round(scd_overall_statistics['std_length'], 13) == \
               round(fst_overall_statistics['std_length'], 13)

        # string_frequency_ should double
        assert len(scd_string_frequency) == len(fst_string_frequency)
        assert np.array_equal(
            list(scd_string_frequency.keys()),
            list(fst_string_frequency.keys())
        )
        for k in fst_string_frequency:
            assert scd_string_frequency[k] == 2 * fst_string_frequency[k]

        # startswith_frequency_ should double
        assert len(scd_startswith_frequency) == len(fst_startswith_frequency)
        assert np.array_equal(
            list(scd_startswith_frequency.keys()),
            list(fst_startswith_frequency.keys())
        )
        for k in fst_startswith_frequency:
            assert scd_startswith_frequency[k] == 2 * fst_startswith_frequency[k]

        # character_frequency_ should double
        assert len(scd_character_frequency) == len(fst_character_frequency)
        assert np.array_equal(
            list(scd_character_frequency.keys()),
            list(fst_character_frequency.keys())
        )
        for k in fst_character_frequency:
            assert scd_character_frequency[k] == 2 * fst_character_frequency[k]



















