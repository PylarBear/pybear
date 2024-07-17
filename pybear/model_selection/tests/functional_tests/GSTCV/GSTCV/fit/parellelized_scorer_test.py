# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pytest

import time
import numpy as np

from model_selection.GSTCV._GSTCV._fit._parallelized_scorer import _parallelized_scorer

from sklearn.metrics import accuracy_score, balanced_accuracy_score

import xgboost as xgb


class TestParallelizedScorer:


    # def _parallelized_scorer(
    #     _X_test: XSKWIPType,
    #     _y_test: YSKWIPType,
    #     _FIT_OUTPUT_TUPLE: [ClassifierProtocol, float, bool],
    #     _f_idx: int,
    #     _SCORER_DICT: ScorerWIPType,
    #     _THRESHOLDS: npt.NDArray[int, float],
    #     _error_score: Union[int, float, None],
    #     _verbose: int,
    #     **scorer_params
    #     ) -> tuple[np.ma.masked_array, np.ma.masked_array]:


    @staticmethod
    @pytest.fixture
    def _X():
        return np.random.randint(0, 10, (100, 10))


    @staticmethod
    @pytest.fixture
    def _y():
        return np.random.randint(0, 2, 100)


    @staticmethod
    @pytest.fixture
    def _fit_output_excepted(_X, _y):

        xgb_clf = xgb.XGBClassifier()
        # [ClassifierProtocol, fit time, fit excepted]
        return (xgb_clf, 0.1, True)


    @staticmethod
    @pytest.fixture
    def _fit_output_good(_X, _y):

        xgb_clf = xgb.XGBClassifier()

        t0 = time.perf_counter()

        xgb_clf.fit(_X[:80], _y[:80])

        tf = time.perf_counter()

        # [ClassifierProtocol, fit time, fit excepted]
        return (xgb_clf, tf-t0, False)



    def test_fit_excepted_accuracy(self, _X, _y, _fit_output_excepted):

        # 5 folds
        _X_test = _X[80:, :]
        _y_test = _y[80:]

        # error_score == np.nan
        out_scores, out_times = _parallelized_scorer(
            _X_test,
            _y_test,
            _FIT_OUTPUT_TUPLE=_fit_output_excepted,
            _f_idx=0,
            _SCORER_DICT={
                'accuracy': accuracy_score,
                'balanced_accuracy': balanced_accuracy_score
            },
            _THRESHOLDS=np.linspace(0,1,21),
            _error_score=np.nan,
            _verbose=10
        )

        assert out_scores.mask.all()
        assert out_times.mask.all()


        # error_score == 0.5 (any arbitrary number)
        out_scores, out_times = _parallelized_scorer(
            _X_test,
            _y_test,
            _FIT_OUTPUT_TUPLE=_fit_output_excepted,
            _f_idx=0,
            _SCORER_DICT={
                'accuracy': accuracy_score,
                'balanced_accuracy': balanced_accuracy_score
            },
            _THRESHOLDS=np.linspace(0,1,21),
            _error_score=0.5,
            _verbose=10
        )

        assert out_scores.mean() == 0.5
        assert out_times.mask.all()


    def test_fit_good_accuracy(self, _X, _y, _fit_output_good):

        # 5 folds
        _X_test = _X[80:, :]
        _y_test = _y[80:]

        # error_score == np.nan
        out_scores, out_times = _parallelized_scorer(
            _X_test,
            _y_test,
            _FIT_OUTPUT_TUPLE=_fit_output_good,
            _f_idx=0,
            _SCORER_DICT={
                'accuracy': accuracy_score,
                'balanced_accuracy': balanced_accuracy_score
            },
            _THRESHOLDS=np.linspace(0, 1, 21),
            _error_score=np.nan,
            _verbose=10
        )

        assert out_scores.shape == (21, 2)
        assert not out_scores.mask.any()
        assert out_scores.min() >= 0
        assert out_scores.max() <= 1
        assert out_scores.mean() > 0

        assert out_times.shape == (21, 2)
        assert not out_times.mask.any()
        assert out_times.min() > 0
        assert out_times.mean() > 0








