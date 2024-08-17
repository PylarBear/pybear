# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pytest

import numpy as np
import dask.array as da
import dask.dataframe as ddf
from distributed import Client
from uuid import uuid4
from sklearn.utils.validation import check_is_fitted
from xgboost import XGBClassifier as sk_XGBClassifier

from model_selection.GSTCV._GSTCVDask.GSTCVDask import GSTCVDask


# post-fit, all attrs should or should not be available based on whether
# data was passed as DF, refit is callable, etc. Lots of ifs, ands, and
# buts

class TestAttrsPostFit:


    @staticmethod
    @pytest.fixture
    def param_grid():
        return {'max_depth': [3,4,5]}


    @staticmethod
    @pytest.fixture
    def columns():
        return [str(uuid4())[:4] for _ in range(3)]


    @staticmethod
    @pytest.fixture
    def cv_int():
        return 4


    @staticmethod
    @pytest.fixture
    def dask_est():

        return sk_XGBClassifier(
            tree_method='hist',
            max_depth=5,
            learning_rate=0.01,
            n_estimators=1000,
            random_state=77
        )


    @staticmethod
    @pytest.fixture
    def dask_GSTCV(dask_est, param_grid):

        def foo(refit=False, scoring=None):

            return GSTCVDask(
                estimator=dask_est,
                param_grid=param_grid,
                cv=4,
                thresholds=[0.5],
                scoring=scoring,
                refit=refit
            )


        return foo


    @staticmethod
    @pytest.fixture
    def X_da_array():
        da.random.seed(2)
        return da.random.randint(0, 10, (100, 3))


    @staticmethod
    @pytest.fixture
    def y_da_array():
        da.random.seed(2)
        return da.random.randint(0, 2, (100,))


    @staticmethod
    @pytest.fixture
    def X_da_DF(X_da_array, columns):
        return ddf.from_dask_array(X_da_array, columns=columns)


    @staticmethod
    @pytest.fixture
    def y_da_DF(y_da_array):
        return ddf.from_dask_array(y_da_array, columns=['y'])


    # exc matches ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
    @staticmethod
    @pytest.fixture(scope='module')
    def generic_no_attribute():
        def foo(_gscv_type, _attr):
            return f"'{_gscv_type}' object has no attribute '{_attr}'"

        return foo


    @staticmethod
    @pytest.fixture(scope='module')
    def generic_not_fitted():
        def foo(_gscv_type, _attr):
            return (f"This {_gscv_type} instance is not fitted yet. Call "
                    f"'fit' with appropriate arguments before using this "
                    f"estimator.")

        return foo
    # END exc matches ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    @staticmethod
    @pytest.fixture(scope='class')
    def _client():
        client = Client(n_workers=None, threads_per_worker=1)
        yield client
        client.close()


    @pytest.mark.parametrize('_format', ('array', 'DF'))
    @pytest.mark.parametrize('_scoring',
        (['balanced_accuracy'], ['accuracy', 'balanced_accuracy'])
    )
    @pytest.mark.parametrize('_refit', (False, 'balanced_accuracy', lambda x: 0))
    def test_dask(self, _refit, _format, _scoring, param_grid, dask_est, cv_int,
        dask_GSTCV, generic_no_attribute, generic_not_fitted, X_da_array, X_da_DF,
        y_da_array, y_da_DF, _client
    ):

        X_dask = X_da_array if _format == 'array' else X_da_DF

        kwargs = {'refit': _refit, 'scoring': _scoring}

        # no DF for y!
        _dask_GSTCV = dask_GSTCV(**kwargs).fit(X_dask, y_da_array)

        # 1a) ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **
        # these are returned no matter what data format is passed or what
        # refit is set to or how many metrics are used ** * ** * ** * **

        __ = getattr(_dask_GSTCV, 'cv_results_')
        assert isinstance(__, dict)  # cv_results is dict
        assert all(map(isinstance, __.keys(), (str for _ in __))) # keys are str
        for _ in __.values():   # values are np masked or np array
            assert(isinstance, ((np.ma.masked_array, np.ndarray)))
        assert len(__[list(__)[0]]) == 3  # number of permutations

        __ = getattr(_dask_GSTCV, 'scorer_')
        assert isinstance(__, dict)   # scorer_ is dict
        assert len(__) == len(_scoring)  # len dict same as len passed
        assert all(map(isinstance, __.keys(), (str for _ in __))) # keys are str
        assert all(map(callable, __.values()))  # keys are callable (sk metrics)

        assert getattr(_dask_GSTCV, 'n_splits_') == cv_int

        # multimetric_ false if 1 scorer, true if 2+ scorers
        assert getattr(_dask_GSTCV, 'multimetric_') is bool(len(_scoring) > 1)

        # END 1a) ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

        # 1b) ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **
        # when there is only one scorer these are returned no matter what
        # data format is passed or what refit is set to but when there
        # is more than one scorer, they are only exposed when refit is
        # not False
        for attr in ('best_params_', 'best_index_'):
            if len(_dask_GSTCV.scorer_) == 1 or \
                len(_dask_GSTCV.scorer_) != 1 and _refit is not False:
                __ = getattr(_dask_GSTCV, attr)
                if attr == 'best_params_':
                    assert isinstance(__, dict)  # best_params_ is dict
                    for param, best_value in __.items():
                        assert param in param_grid  # all keys are in param_grid
                        assert best_value in param_grid[param]  # best value was in grid
                elif attr == 'best_index_':
                    assert int(__) == __  # best_index is integer
                    if isinstance(_refit, str):
                        # if refit is str, returned index is rank 1 in cv_results
                        suffix = 'score' if len(_scoring) == 1 else f'{_refit}'
                        assert _dask_GSTCV.cv_results_[f'rank_test_{suffix}'][__] == 1
                    elif callable(_refit):
                        # if refit is callable, passing cv_results to it == best_idx
                        assert _dask_GSTCV._refit(_dask_GSTCV.cv_results_) == __
                else:
                    raise Exception(f"bad param")
            else:
                with pytest.raises(
                        AttributeError,
                        match=generic_no_attribute('GSTCVDask', attr)
                ):
                    getattr(_dask_GSTCV, attr)

        # END 1b) ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


        # 2a)
        # otherwise these always give attr error when refit is False ** *
        if _refit is False:
            for attr in ('best_estimator_', 'refit_time_', 'classes_',
                'n_features_in_', 'feature_names_in_'
            ):

                # can you guess which kid is doing his own thing
                if attr == 'classes_':

                    exc_info = lambda x: (f"This {x} instance was initialized "
                        f"with `refit=False`. classes_ is available only after "
                        "refitting on the best parameters."
                    )

                    with pytest.raises(
                        AttributeError,
                        match=exc_info('GSTCVDask')
                    ):
                        getattr(_dask_GSTCV, 'classes_')

                else:
                    with pytest.raises(
                        AttributeError,
                        match=generic_no_attribute('GSTCVDask', attr)
                    ):
                        getattr(_dask_GSTCV, attr)
        # END 2a) ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

        # 2b) best_score_ with refit=False: available when there is one
        # scorer, unavailable with multiple ** * ** * ** * ** * ** * ** * ** *
            if len(_dask_GSTCV.scorer_) == 1:
                __ = getattr(_dask_GSTCV, 'best_score_')
                assert __ >= 0
                assert __ <= 1

            else:
                with pytest.raises(
                        AttributeError,
                        match=generic_no_attribute('GSTCVDask', attr)
                ):
                    getattr(_dask_GSTCV, attr)
        # END 2b) ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


        # 3)
        # otherwise, refit is not false and these always return numbers, class
        # instances, or dicts that can use 'isinstance' or '==' ** * ** *

        elif _refit == 'balanced_accuracy' or callable(_refit):

            __ = getattr(_dask_GSTCV, 'best_estimator_')
            assert isinstance(__, type(dask_est))  # same class as given estimator
            assert check_is_fitted(__) is None  # is fitted; otherwise raises
            for param, best_value in _dask_GSTCV.best_params_.items():
                # param values in best_params_ are the param values assigned
                # in best estimator
                assert getattr(__, param) == best_value

            __ = getattr(_dask_GSTCV, 'refit_time_')
            assert isinstance(__, float)
            assert __ > 0

            assert getattr(_dask_GSTCV, 'n_features_in_') == X_dask.shape[1]
        # END otherwise, refit is not false and these always return numbers,
        # class instances, or dicts that can use 'isinstance' or '==' ** * ** *

        # 4a)
        # when refit not False, data format is anything, returns array-like ** *
            __ = getattr(_dask_GSTCV, 'classes_')
            assert isinstance(__, np.ndarray)
            assert np.array_equiv(
                sorted(__),
                sorted(da.unique(y_da_array).compute())
            )
        # END when refit not False, data format is anything, returns array-like ** *

        # 4b)
        # when refit not False, and it matters what the data format is,
        # returns array-like that needs np.array_equiv ** * ** * ** * ** * **
            # feature_names_in_ gives AttrErr when X was array
            if _format == 'array':

                with pytest.raises(
                    AttributeError,
                    match=generic_no_attribute('GSTCVDask', 'feature_names_in_')
                ):
                    getattr(_dask_GSTCV, 'feature_names_in_')

            # feature_names_in_ gives np vector when X was DF
            elif _format == 'DF':
                __ = getattr(_dask_GSTCV, 'feature_names_in_')
                assert isinstance(__, np.ndarray)
                assert len(__) == X_da_array.shape[1]

        # END when refit not False, and it matters what the data format is,
        # returns array-like that needs np.array_equiv ** * ** * ** * ** * **

        # best_score_. this one is crazy.
        # 5a) ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
        # when refit is not False and not a callable, no matter how many
        # scorers there are, dask_GSCV and GSTCVDask return a numeric best_score_.
            if isinstance(_refit, str):
                __ = getattr(_dask_GSTCV, 'best_score_')
                assert isinstance(__, float)
                assert __ >= 0
                assert __ <= 1

                col = f'mean_test_' + ('score' if len(_scoring) == 1 else _refit)
                assert __ == _dask_GSTCV.cv_results_[col][_dask_GSTCV.best_index_]

        # END 5a ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

        # 5b) ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
        # GSTCVDask: when _refit is a callable, if there is only one scorer,
        # GSTCVDask returns a numeric best_score_
            elif callable(_refit):
                if len(_dask_GSTCV.scorer_) == 1:
                    __ = getattr(_dask_GSTCV, 'best_score_')
                    assert isinstance(__, float)
                    assert __ >= 0
                    assert __ <= 1

                    col = f'mean_test_' + ('score' if len(_scoring) == 1 else _refit)
                    assert __ == _dask_GSTCV.cv_results_[col][_dask_GSTCV.best_index_]
        # END 5b) ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

        # 5c) ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **
        # GSTCVDask: when refit is a callable, if there is more than one
        # scorer, GSTCVDask raises AttErr
                else:
                    with pytest.raises(
                        AttributeError,
                        match=generic_no_attribute('GSTCVDask', 'best_score_')
                    ):
                        getattr(_dask_GSTCV, 'best_score_')
        # END 5c) ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

            else:
                raise Exception(f"unexpected refit '{_refit}'")

        # 6) ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
        # GSTCVDask: best_threshold_ is available whenever there is only one
        #   scorer. when multiple scorers, best_threshold_ is only available
        #   when refit is str.

        if len(_dask_GSTCV.scorer_) == 1:
            __ = getattr(_dask_GSTCV, 'best_threshold_')
            assert isinstance(__, float)
            assert __ >= 0
            assert __ <= 1
            _best_idx = _dask_GSTCV.best_index_
            assert _dask_GSTCV.cv_results_[f'best_threshold'][_best_idx] == __
        elif isinstance(_refit, str):
            __ = getattr(_dask_GSTCV, 'best_threshold_')
            assert isinstance(__, float)
            assert __ >= 0
            assert __ <= 1
            _best_thr = \
                lambda col: _dask_GSTCV.cv_results_[col][_dask_GSTCV.best_index_]
            if len(_scoring) == 1:
                assert _best_thr(f'best_threshold') == __
            elif len(_scoring) > 1:
                assert _best_thr(f'best_threshold_{_refit}') == __
        else:
            with pytest.raises(
                AttributeError,
                match=generic_no_attribute('GSTCVDask', 'best_threshold_')
            ):
                getattr(_dask_GSTCV, 'best_threshold_')

        # END 6) ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

















