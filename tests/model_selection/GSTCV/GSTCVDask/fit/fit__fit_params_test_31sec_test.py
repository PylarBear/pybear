# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

import dask.array as da

from sklearn.linear_model import LogisticRegression as sk_LogisticRegression



# def __init__(
#     self,
#     estimator,
#     param_grid: ParamGridType,
#     *,
#     thresholds: Optional[Union[Iterable[Union[int, float]], int, float, None]]=None,
#     scoring: Optional[ScorerInputType]='accuracy',
#     n_jobs: Optional[Union[int, None]]=None,
#     refit: Optional[RefitType]=True,
#     cv: Optional[Union[int, Iterable, None]]=None,
#     verbose: Optional[Union[int, float, bool]]=0,
#     error_score: Optional[Union[Literal['raise'], int, float]]='raise',
#     return_train_score: Optional[bool]=False
# ):



# as of 24_08_31, dask_ml logistic fit() does not accept any fit params.
# for the purpose of this test, replace the dask_ml logistic in the
# session fixture with sklearn


# with Client is faster



class TestCoreFit_FitParams:


    @staticmethod
    @pytest.fixture(scope='function')
    def sk_est_log():

        return sk_LogisticRegression(
            C=1e-5,
            solver='lbfgs',
            n_jobs=1,    # leave this set a 1 because of confliction
            max_iter=100,
            fit_intercept=False
        )



    def test_rejects_sample_weight_too_short(
        self, dask_GSTCV_est_log_one_scorer_prefit, sk_est_log,
        X_da, y_da, _rows, _client
    ):

        # make a copy of dask_GSTCV_, because it is session fixture
        __ = dask_GSTCV_est_log_one_scorer_prefit
        _GSTCV_prefit = type(__)(**__.get_params(deep=False))
        _GSTCV_prefit.set_params(estimator=sk_est_log)

        short_sample_weight = da.random.uniform(0, 1, _rows//2)

        # ValueError should raise inside _parallel_fit ('error_score'=='raise')
        with pytest.raises(ValueError):
            _GSTCV_prefit.fit(X_da, y_da, sample_weight=short_sample_weight)


    def test_rejects_sample_weight_too_long(
        self, dask_GSTCV_est_log_one_scorer_prefit, sk_est_log,
        X_da, y_da, _rows, _client
    ):

        # make a copy of dask_GSTCV_, because it is session fixture
        __ = dask_GSTCV_est_log_one_scorer_prefit
        _GSTCV_prefit = type(__)(**__.get_params(deep=False))
        _GSTCV_prefit.set_params(estimator=sk_est_log)

        long_sample_weight = da.random.uniform(0, 1, _rows*2)

        # ValueError should raise inside _parallel_fit ('error_score'=='raise')
        with pytest.raises(Exception):
            _GSTCV_prefit.fit(X_da, y_da, sample_weight=long_sample_weight)


    def test_correct_sample_weight_works(
        self, dask_GSTCV_est_log_one_scorer_prefit, sk_est_log,
        X_da, y_da, _rows, _client
    ):

        # make a copy of dask_GSTCV_, because it is session fixture
        __ = dask_GSTCV_est_log_one_scorer_prefit
        _GSTCV_prefit = type(__)(**__.get_params(deep=False))
        _GSTCV_prefit.set_params(estimator=sk_est_log)

        correct_sample_weight = da.random.uniform(0, 1, _rows)

        out = _GSTCV_prefit.fit(X_da, y_da, sample_weight=correct_sample_weight)

        assert isinstance(out, type(_GSTCV_prefit))








