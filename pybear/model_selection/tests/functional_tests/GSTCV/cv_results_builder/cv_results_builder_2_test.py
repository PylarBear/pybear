# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

import pytest

import numpy as np

from model_selection.GSTCV._cv_results_builder. \
    gstcv_cv_results_builder_2 import cv_results_builder







class TestCVResultsBuilderTest:

    @staticmethod
    @pytest.fixture
    def param_grid():
        return [
            {'kernel': ['rbf'], 'gamma': [0.1, 0.2], 'test_param': [1, 2, 3]},
            {'kernel': ['poly'], 'degree': [2, 3], 'test_param': [1, 2, 3]},
        ]


    @staticmethod
    @pytest.fixture
    def correct_cv_results_len(param_grid):
        return np.sum(list(map(
            np.prod,
            [[len(_) for _ in __] for __ in map(dict.values, param_grid)]
        )))



    @pytest.mark.parametrize('_cv', (3, 4, 5))
    @pytest.mark.parametrize('_scoring',
        (['accuracy'], ['accuracy', 'balanced_accuracy'])
    )
    @pytest.mark.parametrize('return_train', (True, False))
    def test_cv_results_builder(self, _cv, _scoring, return_train, param_grid,
                                correct_cv_results_len):

        # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
        # BUILD VERIFICATION STUFF ** * ** * ** * ** * ** * ** * ** * ** * ** *
        COLUMN_CHECK = [
            'mean_fit_time',
            'std_fit_time',
            'mean_score_time',
            'std_score_time'
        ]

        # UNIQUE PARAMS
        # notice set!
        COLUMN_CHECK += list({'param_' + _ for __ in param_grid for _ in __})
        COLUMN_CHECK += ['params']
        for sub_scoring in _scoring:

            if len(_scoring) == 1:
                COLUMN_CHECK += ['best_threshold']
            else:
                COLUMN_CHECK += [f'best_threshold_{sub_scoring}']

            suffix = 'score' if len(_scoring) == 1 else sub_scoring
            for split in range(_cv):
                COLUMN_CHECK += [f'split{split}_test_{suffix}']
            COLUMN_CHECK += [f'mean_test_{suffix}']
            COLUMN_CHECK += [f'std_test_{suffix}']
            COLUMN_CHECK += [f'rank_test_{suffix}']
            if return_train:
                for split in range(_cv):
                    COLUMN_CHECK += [f'split{split}_train_{suffix}']
                COLUMN_CHECK += [f'mean_train_{suffix}']
                COLUMN_CHECK += [f'std_train_{suffix}']

        # END BUILD VERIFICATION STUFF ** * ** * ** * ** * ** * ** * ** * ** *
        # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

        # RUN cv_results_builder AND GET CHARACTERISTICS ######################
        cv_results_output, PARAM_GRID_KEY = cv_results_builder(
            param_grid,
            _cv,
            _scoring,
            return_train
        )

        OUTPUT_COLUMNS = list(cv_results_output.keys())
        OUTPUT_LEN = len(cv_results_output['mean_fit_time'])
        # RUN cv_results_builder AND GET CHARACTERISTICS ######################

        # COMPARE OUTPUT TO CONTROLS ##########################################
        for out_col in OUTPUT_COLUMNS:
            assert out_col in COLUMN_CHECK, \
                (f"{out_col} is in OUTPUT_COLUMNS but not in COLUMN_CHECK")

        for check_col in COLUMN_CHECK:
            assert check_col in OUTPUT_COLUMNS, \
                (f"{check_col} is in COLUMN_CHECK but not in OUTPUT_COLUMNS")

        assert OUTPUT_LEN == correct_cv_results_len, \
            (f"output rows ({OUTPUT_LEN}) does not equal expected "
             f"rows ({correct_cv_results_len})")


        assert isinstance(PARAM_GRID_KEY, np.ndarray)
        assert all(map(isinstance, PARAM_GRID_KEY, (np.uint8 for _ in PARAM_GRID_KEY)))
        assert len(PARAM_GRID_KEY) == OUTPUT_LEN

        # COMPARE OUTPUT TO CONTROLS ##########################################























