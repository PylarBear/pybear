# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

import pytest

import numpy as np
import pandas as pd

from model_selection.GSTCV._cv_results_builder._cv_results_builder import cv_results_builder

from model_selection.GSTCV._master_scorer_dict import master_scorer_list





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






    @pytest.mark.parametrize(
        '_cv_results_template',
        [{
            '_n_splits':3,
            '_n_rows':200,
            '_scorer_names':['accuracy', 'balanced_accuracy'],
            '_grids': [{'param_1':[1,2,3], 'param_2':[True, False]}],
            '_return_train_score': True,
            '_fill_param_columns': False
        }],
        indirect=True
    )
    def test_builder_against_template_1(self, _cv_results_template):

        template_pd = pd.DataFrame(_cv_results_template)

        _scorer = {}
        for k, v in master_scorer_list.items():
            if k in ['accuracy', 'balanced_accuracy']:
                _scorer[k] = v

        cv_results_output, _ = cv_results_builder(
            param_grid=[{'param_1':[1,2,3], 'param_2':[True, False]}],
            cv=3,
            scorer=_scorer,
            return_train_score=True
        )

        del _

        cv_results_pd = pd.DataFrame(cv_results_output)

        assert len(cv_results_pd.columns)== len(template_pd.columns)

        assert np.array_equiv(
            sorted(cv_results_pd.columns), sorted(template_pd.columns)
        )




    @pytest.mark.parametrize(
        '_cv_results_template',
        [{
            '_n_splits': 5,
            '_n_rows': 271,
            '_scorer_names': ['balanced_accuracy'],
            '_grids': [{'abc': [1, 2]}, {'xyz': ['a', 'b']}],
            '_return_train_score': False
        }],
        indirect = True
    )
    def test_builder_against_template_2(self, _cv_results_template):

        template_pd = pd.DataFrame(_cv_results_template)

        _scorer = {'balanced_accuracy': master_scorer_list['balanced_accuracy']}

        cv_results_output, _ = cv_results_builder(
            param_grid=[{'abc': [1, 2]}, {'xyz': ['a', 'b']}],
            cv=5,
            scorer=_scorer,
            return_train_score=False
        )

        del _

        cv_results_pd = pd.DataFrame(cv_results_output)

        assert len(cv_results_pd.columns)== len(template_pd.columns)

        assert np.array_equiv(
            sorted(cv_results_pd.columns), sorted(template_pd.columns)
        )









