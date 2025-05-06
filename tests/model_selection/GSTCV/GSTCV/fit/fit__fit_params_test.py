# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

import numpy as np



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



class TestCoreFit_FitParams_NotPipe:


    def test_rejects_sample_weight_too_short(
        self, sk_GSTCV_est_log_one_scorer_prefit, X_np, y_np, _rows
    ):

        # make a copy of sk_GSTCV, because it is session fixture
        __ = sk_GSTCV_est_log_one_scorer_prefit
        _GSTCV_prefit = type(__)(**__.get_params(deep=False))

        short_sample_weight = np.random.uniform(0, 1, _rows//2)

        # ValueError should raise inside _parallel_fit ('error_score'=='raise')
        with pytest.raises(ValueError):
            _GSTCV_prefit.fit(X_np, y_np, sample_weight=short_sample_weight)


    def test_rejects_sample_weight_too_long(
        self, sk_GSTCV_est_log_one_scorer_prefit, X_np, y_np, _rows
    ):

        # make a copy of sk_GSTCV, because it is session fixture
        __ = sk_GSTCV_est_log_one_scorer_prefit
        _GSTCV_prefit = type(__)(**__.get_params(deep=False))

        long_sample_weight = np.random.uniform(0, 1, _rows*2)

        # ValueError should raise inside _parallel_fit ('error_score'=='raise')
        with pytest.raises(ValueError):
            _GSTCV_prefit.fit(X_np, y_np, sample_weight=long_sample_weight)


    def test_correct_sample_weight_works(
            self, sk_GSTCV_est_log_one_scorer_prefit, X_np, y_np, _rows
    ):

        # make a copy of est, because sk_GSTCV_ is session fixture
        __ = sk_GSTCV_est_log_one_scorer_prefit
        _GSTCV_prefit = type(__)(**__.get_params(deep=False))

        correct_sample_weight = np.random.uniform(0, 1, _rows)

        out = _GSTCV_prefit.fit(X_np, y_np, sample_weight=correct_sample_weight)

        assert isinstance(out, type(_GSTCV_prefit))



class TestCoreFit_FitParams_Pipe:


    def test_rejects_sample_weight_too_short(
        self, sk_GSTCV_pipe_log_one_scorer_prefit, X_np, y_np, _rows
    ):

        # make a copy of sk_GSTCV, because it is session fixture
        __ = sk_GSTCV_pipe_log_one_scorer_prefit
        _GSTCV_PIPE_prefit = type(__)(**__.get_params(deep=False))

        short_sample_weight = np.random.uniform(0, 1, _rows//2)

        # ValueError should raise inside _parallel_fit ('error_score'=='raise')
        with pytest.raises(ValueError):
            _GSTCV_PIPE_prefit.fit(
                X_np, y_np, sk_logistic__sample_weight=short_sample_weight
            )


    def test_rejects_sample_weight_too_long(
        self, sk_GSTCV_pipe_log_one_scorer_prefit, X_np, y_np, _rows
    ):

        # make a copy of sk_GSTCV, because it is session fixture
        __ = sk_GSTCV_pipe_log_one_scorer_prefit
        _GSTCV_PIPE_prefit = type(__)(**__.get_params(deep=False))

        long_sample_weight = np.random.uniform(0, 1, _rows*2)

        # ValueError should raise inside _parallel_fit ('error_score'=='raise')
        with pytest.raises(ValueError):
            _GSTCV_PIPE_prefit.fit(
                X_np, y_np, sk_logistic__sample_weight=long_sample_weight
            )


    def test_correct_sample_weight_works(
            self, sk_GSTCV_pipe_log_one_scorer_prefit, X_np, y_np, _rows
    ):

        # make a copy of sk_GSTCV, because it is session fixture
        __ = sk_GSTCV_pipe_log_one_scorer_prefit
        _GSTCV_PIPE_prefit = type(__)(**__.get_params(deep=False))

        correct_sample_weight = np.random.uniform(0, 1, _rows)

        out = _GSTCV_PIPE_prefit.fit(
            X_np, y_np, sk_logistic__sample_weight=correct_sample_weight
        )

        assert isinstance(out, type(_GSTCV_PIPE_prefit))








