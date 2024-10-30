# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pytest

import numpy as np
import dask.array as da
from sklearn.utils.validation import check_is_fitted


# post-fit, all attrs should or should not be available based on whether
# data was passed as DF, refit is callable, etc. Lots of ifs, ands, and
# buts

pytest.skip(reason=f'pipes take too long', allow_module_level=True)

class TestDaskAttrsPostFit:


    # dont use client, too slow 24_08_26

    @pytest.mark.parametrize('_format', ('array', 'DF'))
    @pytest.mark.parametrize('_scoring',
        (['accuracy'], ['accuracy', 'balanced_accuracy'])
    )
    @pytest.mark.parametrize('_refit', (False, 'accuracy', lambda x: 0))
    def test_dask(self, _refit, _format, _scoring, param_grid_pipe_dask_log,
        dask_pipe_log, standard_cv_int, _refit_false, generic_no_attribute_1,
        X_da, X_ddf, y_da, y_ddf, _cols, COLUMNS,
        dask_GSTCV_pipe_log_one_scorer_postfit_refit_false_fit_on_da,
        dask_GSTCV_pipe_log_one_scorer_postfit_refit_str_fit_on_da,
        dask_GSTCV_pipe_log_one_scorer_postfit_refit_fxn_fit_on_da,
        dask_GSTCV_pipe_log_two_scorers_postfit_refit_false_fit_on_da,
        dask_GSTCV_pipe_log_two_scorers_postfit_refit_str_fit_on_da,
        dask_GSTCV_pipe_log_two_scorers_postfit_refit_fxn_fit_on_da,
        dask_GSTCV_pipe_log_one_scorer_postfit_refit_false_fit_on_ddf,
        dask_GSTCV_pipe_log_one_scorer_postfit_refit_str_fit_on_ddf,
        dask_GSTCV_pipe_log_one_scorer_postfit_refit_fxn_fit_on_ddf,
        dask_GSTCV_pipe_log_two_scorers_postfit_refit_false_fit_on_ddf,
        dask_GSTCV_pipe_log_two_scorers_postfit_refit_str_fit_on_ddf,
        dask_GSTCV_pipe_log_two_scorers_postfit_refit_fxn_fit_on_ddf,
        # _client
    ):

        if _format == 'array':
            if _scoring == ['accuracy']:
                if _refit is False:
                    _dask_GSTCV_PIPE = \
                        dask_GSTCV_pipe_log_one_scorer_postfit_refit_false_fit_on_da
                elif _refit == 'accuracy':
                    _dask_GSTCV_PIPE = \
                        dask_GSTCV_pipe_log_one_scorer_postfit_refit_str_fit_on_da
                elif callable(_refit):
                    _dask_GSTCV_PIPE = \
                        dask_GSTCV_pipe_log_one_scorer_postfit_refit_fxn_fit_on_da

            elif _scoring == ['accuracy', 'balanced_accuracy']:
                if _refit is False:
                    _dask_GSTCV_PIPE = \
                        dask_GSTCV_pipe_log_two_scorers_postfit_refit_false_fit_on_da
                elif _refit == 'accuracy':
                    _dask_GSTCV_PIPE = \
                        dask_GSTCV_pipe_log_two_scorers_postfit_refit_str_fit_on_da
                elif callable(_refit):
                    _dask_GSTCV_PIPE = \
                        dask_GSTCV_pipe_log_two_scorers_postfit_refit_fxn_fit_on_da

        elif _format == 'DF':
            if _scoring == ['accuracy']:
                if _refit is False:
                    _dask_GSTCV_PIPE = \
                        dask_GSTCV_pipe_log_one_scorer_postfit_refit_false_fit_on_ddf
                elif _refit == 'accuracy':
                    _dask_GSTCV_PIPE = \
                        dask_GSTCV_pipe_log_one_scorer_postfit_refit_str_fit_on_ddf
                elif callable(_refit):
                    _dask_GSTCV_PIPE = \
                        dask_GSTCV_pipe_log_one_scorer_postfit_refit_fxn_fit_on_ddf

            elif _scoring == ['accuracy', 'balanced_accuracy']:
                if _refit is False:
                    _dask_GSTCV_PIPE = \
                        dask_GSTCV_pipe_log_two_scorers_postfit_refit_false_fit_on_ddf
                elif _refit == 'accuracy':
                    _dask_GSTCV_PIPE = \
                        dask_GSTCV_pipe_log_two_scorers_postfit_refit_str_fit_on_ddf
                elif callable(_refit):
                    _dask_GSTCV_PIPE = \
                        dask_GSTCV_pipe_log_two_scorers_postfit_refit_fxn_fit_on_ddf

        # 1a) ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **
        # these are returned no matter what data format is passed or what
        # refit is set to or how many metrics are used ** * ** * ** * **

        __ = getattr(_dask_GSTCV_PIPE, 'cv_results_')
        assert isinstance(__, dict)  # cv_results is dict
        assert all(map(isinstance, __.keys(), (str for _ in __))) # keys are str
        for _ in __.values():   # values are np masked or np array
            assert isinstance(_, (np.ma.masked_array, np.ndarray))
        assert len(__[list(__)[0]]) == 4  # number of permutations

        __ = getattr(_dask_GSTCV_PIPE, 'scorer_')
        assert isinstance(__, dict)   # scorer_ is dict
        assert len(__) == len(_scoring)  # len dict same as len passed
        assert all(map(isinstance, __.keys(), (str for _ in __))) # keys are str
        assert all(map(callable, __.values()))  # keys are callable (sk metrics)

        assert getattr(_dask_GSTCV_PIPE, 'n_splits_') == standard_cv_int

        # multimetric_ false if 1 scorer, true if 2+ scorers
        assert getattr(_dask_GSTCV_PIPE, 'multimetric_') is bool(len(_scoring) > 1)

        # END 1a) ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

        # 1b) ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **
        # when there is only one scorer these are returned no matter what
        # data format is passed or what refit is set to but when there
        # is more than one scorer, they are only exposed when refit is
        # not False
        for attr in ('best_params_', 'best_index_'):
            if len(_dask_GSTCV_PIPE.scorer_) == 1 or \
                len(_dask_GSTCV_PIPE.scorer_) != 1 and _refit is not False:
                __ = getattr(_dask_GSTCV_PIPE, attr)
                if attr == 'best_params_':
                    assert isinstance(__, dict)  # best_params_ is dict
                    for param, best_value in __.items():
                        # all keys are in param_grid
                        assert param in param_grid_pipe_dask_log
                        # best value was in grid
                        assert best_value in param_grid_pipe_dask_log[param]
                elif attr == 'best_index_':
                    assert int(__) == __  # best_index is integer
                    if isinstance(_refit, str):
                        # if refit is str, returned index is rank 1 in cv_results
                        suffix = 'score' if len(_scoring) == 1 else f'{_refit}'
                        col = f'rank_test_{suffix}'
                        assert _dask_GSTCV_PIPE.cv_results_[col][__] == 1
                    elif callable(_refit):
                        # if refit is callable, passing cv_results to it == best_idx
                        assert __ == \
                            _dask_GSTCV_PIPE._refit(_dask_GSTCV_PIPE.cv_results_)
                else:
                    raise Exception(f"bad param")
            else:
                with pytest.raises(
                        AttributeError,
                        match=generic_no_attribute_1('GSTCVDask', attr)
                ):
                    getattr(_dask_GSTCV_PIPE, attr)

        # END 1b) ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


        # 2a)
        # otherwise these always give attr error when refit is False ** *
        if _refit is False:
            for attr in ('best_estimator_', 'refit_time_', 'classes_',
                'n_features_in_', 'feature_names_in_'
            ):

                # can you guess which kid is doing his own thing
                if attr == 'classes_':

                    with pytest.raises(
                        AttributeError,
                        match=_refit_false('GSTCVDask')
                    ):
                        getattr(_dask_GSTCV_PIPE, 'classes_')

                else:
                    with pytest.raises(
                        AttributeError,
                        match=generic_no_attribute_1('GSTCVDask', attr)
                    ):
                        getattr(_dask_GSTCV_PIPE, attr)
        # END 2a) ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

        # 2b) best_score_ with refit=False: available when there is one
        # scorer, unavailable with multiple ** * ** * ** * ** * ** * ** * ** *
            if len(_dask_GSTCV_PIPE.scorer_) == 1:
                __ = getattr(_dask_GSTCV_PIPE, 'best_score_')
                assert __ >= 0
                assert __ <= 1
            else:
                with pytest.raises(
                        AttributeError,
                        match=generic_no_attribute_1('GSTCVDask', attr)
                ):
                    getattr(_dask_GSTCV_PIPE, attr)
        # END 2b) ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


        # 3)
        # otherwise, refit is not false and these always return numbers, class
        # instances, or dicts that can use 'isinstance' or '==' ** * ** *

        elif _refit == 'accuracy' or callable(_refit):

            __ = getattr(_dask_GSTCV_PIPE, 'best_estimator_')
                                # same class as given estimator
            assert isinstance(__, type(dask_pipe_log))
            assert check_is_fitted(__) is None  # is fitted; otherwise cif() raises

            og_params = _dask_GSTCV_PIPE.estimator.get_params(deep=True)

            for param, value in __.get_params(deep=True).items():

                # if param from best_estimator_.get_params is not in best_params_,
                # it should be equal to the value given in the originally-
                # passed estimator (except for the transformers/estimators)
                if param not in _dask_GSTCV_PIPE.best_params_:
                    if value is np.nan:
                        assert og_params[param] is np.nan
                    # best_estimator_ transformers/estimators most likely
                    # have different hyperparam values than the og estimator,
                    # just verify isinstance
                    elif param == 'steps':
                        assert isinstance(value[0][1], type(og_params[param][0][1]))
                        assert isinstance(value[1][1], type(og_params[param][1][1]))
                    elif param == 'dask_StandardScaler':
                        assert isinstance(value, type(og_params[param]))
                    elif param == 'dask_logistic':
                        assert isinstance(value, type(og_params[param]))
                    else:
                        assert value == og_params[param]
                # if the best_estimator_.param is in best_params_, the value
                # for best_estimator_ should equal the value in best_params_
                elif param in _dask_GSTCV_PIPE.best_params_:
                    if value is np.nan:
                        assert _dask_GSTCV_PIPE.best_params_[param] is np.nan
                    else:
                        assert value == _dask_GSTCV_PIPE.best_params_[param]
                else:
                    raise Exception(f"param not correctly assigned")

            __ = getattr(_dask_GSTCV_PIPE, 'refit_time_')
            assert isinstance(__, float)
            assert __ > 0

            assert getattr(_dask_GSTCV_PIPE, 'n_features_in_') == _cols
        # END otherwise, refit is not false and these always return numbers,
        # class instances, or dicts that can use 'isinstance' or '==' ** * ** *

        # 4a)
        # when refit not False, data format is anything, returns array-like ** *
            __ = getattr(_dask_GSTCV_PIPE, 'classes_')
            assert isinstance(__, np.ndarray)
            assert np.array_equiv(
                sorted(__),
                sorted(da.unique(y_da).compute())
            )
        # END when refit not False, data format is anything, returns array-like ** *

        # 4b)
        # when refit not False, and it matters what the data format is,
        # returns array-like that needs np.array_equiv ** * ** * ** * ** * **
            # feature_names_in_ gives AttrErr when X was array
            if _format == 'array':

                with pytest.raises(
                    AttributeError,
                    match=generic_no_attribute_1('GSTCVDask', 'feature_names_in_')
                ):
                    getattr(_dask_GSTCV_PIPE, 'feature_names_in_')

            # feature_names_in_ gives np vector when X was DF
            elif _format == 'DF':
                __ = getattr(_dask_GSTCV_PIPE, 'feature_names_in_')
                assert isinstance(__, np.ndarray)
                assert np.array_equiv(__, COLUMNS)

        # END when refit not False, and it matters what the data format is,
        # returns array-like that needs np.array_equiv ** * ** * ** * ** * **

        # best_score_. this one is crazy.
        # 5a) ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
        # when refit is not False and not a callable, no matter how many
        # scorers there are, dask_GSCV and GSTCVDask return a numeric best_score_.
            if isinstance(_refit, str):
                __ = getattr(_dask_GSTCV_PIPE, 'best_score_')
                assert isinstance(__, float)
                assert __ >= 0
                assert __ <= 1

                col = f'mean_test_' + ('score' if len(_scoring) == 1 else _refit)
                assert __ == \
                    _dask_GSTCV_PIPE.cv_results_[col][_dask_GSTCV_PIPE.best_index_]

        # END 5a ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

        # 5b) ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
        # GSTCVDask: when _refit is a callable, if there is only one scorer,
        # GSTCVDask returns a numeric best_score_
            elif callable(_refit):
                if len(_dask_GSTCV_PIPE.scorer_) == 1:
                    __ = getattr(_dask_GSTCV_PIPE, 'best_score_')
                    assert isinstance(__, float)
                    assert __ >= 0
                    assert __ <= 1

                    col = f'mean_test_' + ('score' if len(_scoring) == 1 else _refit)
                    assert __ == \
                        _dask_GSTCV_PIPE.cv_results_[col][_dask_GSTCV_PIPE.best_index_]
        # END 5b) ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

        # 5c) ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **
        # GSTCVDask: when refit is a callable, if there is more than one
        # scorer, GSTCVDask raises AttErr
                else:
                    with pytest.raises(
                        AttributeError,
                        match=generic_no_attribute_1('GSTCVDask', 'best_score_')
                    ):
                        getattr(_dask_GSTCV_PIPE, 'best_score_')
        # END 5c) ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

            else:
                raise Exception(f"unexpected refit '{_refit}'")

        # 6) ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
        # GSTCVDask: best_threshold_ is available whenever there is only one
        #   scorer. when multiple scorers, best_threshold_ is only available
        #   when refit is str.

        if len(_dask_GSTCV_PIPE.scorer_) == 1:
            __ = getattr(_dask_GSTCV_PIPE, 'best_threshold_')
            assert isinstance(__, float)
            assert __ >= 0
            assert __ <= 1
            _best_idx = _dask_GSTCV_PIPE.best_index_
            assert _dask_GSTCV_PIPE.cv_results_[f'best_threshold'][_best_idx] == __
        elif isinstance(_refit, str):
            __ = getattr(_dask_GSTCV_PIPE, 'best_threshold_')
            assert isinstance(__, float)
            assert __ >= 0
            assert __ <= 1
            _best_thr = lambda col: (
                _dask_GSTCV_PIPE.cv_results_[col][_dask_GSTCV_PIPE.best_index_]
            )
            if len(_scoring) == 1:
                assert _best_thr(f'best_threshold') == __
            elif len(_scoring) > 1:
                assert _best_thr(f'best_threshold_{_refit}') == __
        else:
            with pytest.raises(
                AttributeError,
                match=generic_no_attribute_1('GSTCVDask', 'best_threshold_')
            ):
                getattr(_dask_GSTCV_PIPE, 'best_threshold_')

        # END 6) ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

















