# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

import pytest

import time
import numpy as np
from model_selection.GSTCV._GSTCV._fit._parallelized_fit import _parallelized_fit



class TestParallelizedFit:

    @staticmethod
    @pytest.fixture
    def mock_X():
        return np.random.randint(0, 10, (20,5))


    @staticmethod
    @pytest.fixture
    def mock_y():
        return np.random.randint(0,2, (20,1))


    @staticmethod
    @pytest.fixture
    def mock_train_idxs(mock_X):
        return list(range(mock_X.shape[0] - 5))


    @staticmethod
    @pytest.fixture
    def mock_test_idxs(mock_X):
        return list(range(mock_X.shape[0] - 5, mock_X.shape[0]))



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

                time.sleep(1)

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
                return np.random.uniform()


        return MockClassifier





    #     def _parallelized_fit(
    #             f_idx: int,
    #             X_train: XType,
    #             y_train: YType,
    #             # train_idxs,
    #             # test_idxs,
    #             _estimator_,
    #             _grid: dict[str, Union[str, int, float, bool]],
    #             _error_score,
    #             **fit_params
    #     ):


    def test_when_completes_fit(self, mock_classifier, mock_X, mock_y,
                                mock_train_idxs, mock_test_idxs):
        # returns fitted est, time, fit_excepted == False, train_idxs, test_idxs
        out_fitted_estimator, out_time, out_fit_excepted, train_idxs, test_idxs = \
            _parallelized_fit(
                np.random.randint(0,10),
                mock_X,
                mock_y,
                train_idxs=mock_train_idxs,
                test_idxs=mock_test_idxs,
                _estimator_=mock_classifier(),
                _grid = {'param_1': True, 'param_2': [3,4,5]},
                _error_score=np.nan,
                # **fit_params
            )

        assert isinstance(out_fitted_estimator.score_, float)
        assert out_fitted_estimator.score_ >= 0
        assert out_fitted_estimator.score_ <= 1
        assert isinstance(out_time, float)
        assert out_time > 1
        assert out_fit_excepted is False
        assert np.array_equiv(train_idxs, mock_train_idxs)
        assert np.array_equiv(test_idxs, mock_test_idxs)



    def test_type_error(self, mock_classifier, mock_X, mock_y,
                        mock_train_idxs, mock_test_idxs):
        # if TypeError, raise TypeError
        with pytest.raises(TypeError):
            _parallelized_fit(
                np.random.randint(0,10),
                mock_X,
                mock_y,
                train_idxs=mock_train_idxs,
                test_idxs=mock_test_idxs,
                _estimator_=mock_classifier(command='type_error'),
                _grid = {'param_1': True, 'param_2': [3,4,5]},
                _error_score=np.nan,
                # **fit_params
            )



    def test_other_error_with_raise(self, mock_classifier, mock_X, mock_y,
                                     mock_train_idxs, mock_test_idxs):
        # if error_score == 'raise', raise Exception
        with pytest.raises(ValueError):
            _parallelized_fit(
                np.random.randint(0,10),
                mock_X,
                mock_y,
                train_idxs=mock_train_idxs,
                test_idxs=mock_test_idxs,
                _estimator_=mock_classifier(command='other_error_with_raise'),
                _grid = {'param_1': True, 'param_2': [3,4,5]},
                _error_score='raise',  # ineffectual
                # **fit_params
            )


    def test_other_error_not_raise(self, mock_classifier, mock_X, mock_y,
                                    mock_train_idxs, mock_test_idxs):
        # else warn, fit_excepted = True
        # returns fitted est, time, fit_excepted == False, train_idx, test_idxs

        out_fitted_estimator, out_time, out_fit_excepted, train_idxs, test_idxs = \
            _parallelized_fit(
                np.random.randint(0,10),
                mock_X,
                mock_y,
                train_idxs=mock_train_idxs,
                test_idxs=mock_test_idxs,
                _estimator_=mock_classifier(command='other_error_not_raise'),
                _grid = {'param_1': True, 'param_2': [3,4,5]},
                _error_score=np.nan,
                # **fit_params
        )

        assert isinstance(out_fitted_estimator.score_, float)
        assert out_fitted_estimator.score_ is np.nan
        assert isinstance(out_time, float)
        assert out_time > 1
        assert out_fit_excepted is True
        assert np.array_equiv(train_idxs, mock_train_idxs)
        assert np.array_equiv(test_idxs, mock_test_idxs)


    def test_fit_params(self, mock_classifier, mock_X, mock_y,
                        mock_train_idxs, mock_test_idxs):

        with pytest.raises(BrokenPipeError):
            _parallelized_fit(
                np.random.randint(0,10),
                mock_X,
                mock_y,
                train_idxs=mock_train_idxs,
                test_idxs=mock_test_idxs,
                _estimator_=mock_classifier(),
                _grid = {'param_1': True, 'param_2': [3,4,5]},
                _error_score=np.nan,
                kill=True
            )






















