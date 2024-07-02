# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

import numpy as np

from model_selection.GSTCV._GSTCVMixin._fit._get_best_thresholds import _get_best_thresholds





class TestGetBestThresholds:

    # def _get_best_thresholds(
    #         _trial_idx: int,
    #         _scorer_names: list[str],
    #         _TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_MATRIX: IntermediateHolderType,
    #         _THRESHOLDS: npt.NDArray[Union[float, int]],
    #         _cv_results: CVResultsType
    # ) -> npt.NDArray[np.uint16]:

    # 5 splits
    # 11 thresholds
    # 4 scorers

    @staticmethod
    @pytest.fixture(scope='class')
    def _TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_MATRIX():

        return np.ma.masked_array(np.random.uniform(0, 1, (5, 11, 4)))


    @staticmethod
    @pytest.fixture(scope='class')
    def _scorer_names():

        return ['accuracy', 'balanced_accuracy', 'precision', 'f1']


    @staticmethod
    @pytest.fixture(scope='class')
    def _THRESHOLDS():

        return np.linspace(0, 1, 11, dtype=np.float64)






    @pytest.mark.parametrize('junk_trial_idx',
        (-1, 3.14, True, min, 'junk', [0,1], (0,1), {0,1}, {'a':1}, lambda x: x)
    )
    def test_rejects_junk_trial_idx(self, junk_trial_idx, _scorer_names,
        _TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_MATRIX, _THRESHOLDS):

        with pytest.raises(TypeError):
            _get_best_thresholds(
                junk_trial_idx,
                _scorer_names,
                _TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_MATRIX,
                _THRESHOLDS,
                {'abcdefghijklm'[_]: np.ma.masked_array() for _ in range(10)}
            )


    @pytest.mark.parametrize('junk_scorer_names',
        (-1, 3.14, True, min, 'junk', [0,1], (0,1), {0,1}, {'a':1}, lambda x: x)
    )
    def test_rejects_junk_scorer_names(self, junk_scorer_names,
        _TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_MATRIX, _THRESHOLDS):

        with pytest.raises(ValueError):
            _get_best_thresholds(
                0,
                junk_scorer_names,
                _TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_MATRIX,
                _THRESHOLDS,
                {'abcdefghijklm'[_]: np.ma.masked_array() for _ in range(10)}
            )


    @pytest.mark.parametrize('junk_holder',
        (-1, 3.14, True, min, 'junk', [0,1], (0,1), {0,1}, {'a':1}, lambda x: x)
    )
    def test_rejects_junk_holder(self, junk_holder, _scorer_names,
        _TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_MATRIX, _THRESHOLDS):

        with pytest.raises(AssertionError):
            _get_best_thresholds(
                0,
                _scorer_names,
                junk_holder,
                _THRESHOLDS,
                {'abcdefghijklm'[_]: np.ma.masked_array() for _ in range(10)}
            )


    @pytest.mark.parametrize('_trial_idx', (0, 3, 9))
    @pytest.mark.parametrize(
        '_cv_results_template',
        [{
            '_n_splits': 5,
            '_n_rows': 15,
            '_scorer_names': ['accuracy', 'balanced_accuracy', 'precision', 'f1'],
            '_grids': [{'abc': [1, 2], 'xyz': ['a', 'b']}],
            '_return_train_score': True,
            '_fill_param_columns': False
        }],
        indirect = True
    )
    def test_accuracy(self, _trial_idx, _scorer_names, _cv_results_template,
        _TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_MATRIX, _THRESHOLDS):

        out = _get_best_thresholds(
                0,
                _scorer_names,
                _TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_MATRIX,
                _THRESHOLDS,
                _cv_results_template
            )


        assert isinstance(out, np.ndarray)
        assert out.dtype == np.uint16
        assert len(out) == len(_scorer_names)
        assert min(out) >= 0
        assert max(out) < len(_THRESHOLDS)























