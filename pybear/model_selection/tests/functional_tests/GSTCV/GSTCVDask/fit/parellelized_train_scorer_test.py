# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pytest

import time
import numpy as np
import dask.array as da


# pizza, when all is said and done, and the GSCV anomalies are explained away,
# if no changes made to dask _parallelized_fit/_scorer/_train_score, delete the
# dask ones, move the SK ones to 'shared_fit' and use them for SK & dask.

from model_selection.GSTCV._GSTCVDask._fit._parallelized_train_scorer import \
    _parallelized_train_scorer

from sklearn.metrics import accuracy_score, balanced_accuracy_score

import xgboost as xgb





class TestParallelizedScorer:


    # def _parallelized_train_scorer(
    #     _X_train: XDaskWIPType,
    #     _y_train: YDaskWIPType,
    #     _FIT_OUTPUT_TUPLE: tuple[ClassifierProtocol, float, bool],
    #     _f_idx: int,
    #     _SCORER_DICT: ScorerWIPType,
    #     _best_threshold: Union[int, float],
    #     _error_score: Union[int, float, None],
    #     _verbose: int,
    #     **scorer_params
    #     ) -> np.ma.masked_array:


    @staticmethod
    @pytest.fixture
    def _X():
        return da.random.randint(0, 10, (100, 10))


    @staticmethod
    @pytest.fixture
    def _y():
        return da.random.randint(0, 2, 100)


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
        _X_train = _X[:80, :]
        _y_train = _y[:80]


        # error_score == np.nan
        out_scores = _parallelized_train_scorer(
            _X_train,
            _y_train,
            _FIT_OUTPUT_TUPLE=_fit_output_excepted,
            _f_idx=0,
            _SCORER_DICT={
                'accuracy': accuracy_score,
                'balanced_accuracy': balanced_accuracy_score
            },
            _BEST_THRESHOLDS_BY_SCORER=np.array([45, 55]),
            _error_score=np.nan,
            _verbose=10
        )

        assert out_scores.mask.all()


        # error_score == 0.5 (any arbitrary number)
        out_scores = _parallelized_train_scorer(
            _X_train,
            _y_train,
            _FIT_OUTPUT_TUPLE=_fit_output_excepted,
            _f_idx=0,
            _SCORER_DICT={
                'accuracy': accuracy_score,
                'balanced_accuracy': balanced_accuracy_score
            },
            _BEST_THRESHOLDS_BY_SCORER=np.array([40, 60]),
            _error_score=0.5,
            _verbose=10
        )

        assert out_scores.mean() == 0.5


    def test_fit_good_accuracy(self, _X, _y, _fit_output_good):

        # 5 folds
        _X_train = _X[:80, :]
        _y_train = _y[:80]

        # error_score == np.nan
        out_scores = _parallelized_train_scorer(
            _X_train,
            _y_train,
            _FIT_OUTPUT_TUPLE=_fit_output_good,
            _f_idx=0,
            _SCORER_DICT={
                'accuracy': accuracy_score,
                'balanced_accuracy': balanced_accuracy_score
            },
            _BEST_THRESHOLDS_BY_SCORER=np.array([48, 52]),
            _error_score=np.nan,
            _verbose=10
        )


        assert out_scores.shape == (2,)
        assert not out_scores.mask.any()
        assert out_scores.min() >= 0
        assert out_scores.max() <= 1
        assert out_scores.mean() > 0







