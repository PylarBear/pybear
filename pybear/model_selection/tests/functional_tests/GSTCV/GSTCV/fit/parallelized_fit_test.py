# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

import pytest

import time

import numpy as np
from pybear.model_selection.GSTCV._GSTCV._fit._parallelized_fit import \
    _parallelized_fit



class TestParallelizedFit:


    @staticmethod
    @pytest.fixture
    def mock_classifier():
        class MockClassifier:

            def __init__(self, command='run'):

                self.command = command
                self.is_fitted = False
                # command can be 'type_error', 'other_error_raise',
                # 'other_error_not_raise', 'run'

            def fit(self, X, y, **fit_params):

                time.sleep(0.5)

                if len(fit_params) and fit_params['kill'] is True:
                    raise BrokenPipeError     # an obscure error

                if self.command == 'run':
                    self.score_ = self.score(X, y)
                    self.is_fitted = True
                elif self.command == 'type_error':
                    raise TypeError
                elif self.command == 'other_error_with_raise':
                    raise TabError # an obscure error
                elif self.command == 'other_error_not_raise':
                    self.score_ = np.nan
                    raise TabError # an obscure error

                return self


            def score(self, X, y):

                return np.random.uniform(0, 1)


        return MockClassifier

    #     def _parallelized_fit(
    #             f_idx: int,
    #             X_train: XSKWIPType,
    #             y_train: YSKWIPType,
    #             _estimator_,
    #             _grid: dict[str, Union[str, int, float, bool]],
    #             _error_score,
    #             **fit_params
    #     ):


    def test_when_completes_fit(self, mock_classifier, X_np, y_np):
        # returns fitted est, time, fit_excepted == False
        out_fitted_estimator, out_time, out_fit_excepted = \
            _parallelized_fit(
                np.random.randint(0,10),  # f_idx
                X_np,
                y_np,
                _estimator_=mock_classifier(),
                _grid = {'param_1': True, 'param_2': [3,4,5]},
                _error_score=np.nan,
                # **fit_params
            )

        assert isinstance(out_fitted_estimator, mock_classifier)
        assert isinstance(out_fitted_estimator.score_, float)
        assert out_fitted_estimator.score_ >= 0
        assert out_fitted_estimator.score_ <= 1
        assert isinstance(out_time, float)
        assert out_time > 0.5
        assert out_fit_excepted is False


    def test_other_error_with_raise(self, mock_classifier, X_np, y_np):
        # if error_score == 'raise', raise Exception
        with pytest.raises(ValueError):
            _parallelized_fit(
                np.random.randint(0,10),  # f_idx
                X_np,
                y_np,
                _estimator_=mock_classifier(command='other_error_with_raise'),
                _grid = {'param_1': True, 'param_2': [3,4,5]},
                _error_score='raise',  # ineffectual
                # **fit_params
            )


    def test_other_error_not_raise(self, mock_classifier, X_np, y_np):
        # else warn, fit_excepted = True
        # returns fitted est, time, fit_excepted == False

        out_fitted_estimator, out_time, out_fit_excepted = \
            _parallelized_fit(
                np.random.randint(0,10),  # f_idx
                X_np,
                y_np,
                _estimator_=mock_classifier(command='other_error_not_raise'),
                _grid = {'param_1': True, 'param_2': [3,4,5]},
                _error_score=np.nan,
                # **fit_params
        )

        assert isinstance(out_fitted_estimator, mock_classifier)
        assert isinstance(out_fitted_estimator.score_, float)
        assert out_fitted_estimator.score_ is np.nan
        assert isinstance(out_time, float)
        assert out_time > 0.5
        assert out_fit_excepted is True


    def test_fit_params(self, mock_classifier, X_np, y_np):

        with pytest.raises(BrokenPipeError):
            _parallelized_fit(
                np.random.randint(0,10),  # f_idx
                X_np,
                y_np,
                _estimator_=mock_classifier(),
                _grid = {'param_1': True, 'param_2': [3,4,5]},
                _error_score=np.nan,
                kill=True
            )






















