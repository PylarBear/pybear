import numpy as np
import pandas as pd
import dask.dataframe as ddf
from dask_ml.datasets import make_classification as da_make_classification
from dask_ml.linear_model import LogisticRegression as dask_LogisticRegression
import string

from model_selection.GSTCV._GSTCVDask.GSTCVDask import GSTCVDask


import pytest


@pytest.skip(reason=f"pizza finish", allow_module_level=True)
class TestGSTCVDask:

    _n_classes = 2
    _num_rows = 1_000
    _num_cols = 20
    _column_names = list(string.ascii_lowercase)[:_num_cols]

    _X_da, _y_da = da_make_classification(
        n_classes=_n_classes,
        n_samples=_num_rows,
        n_features=_num_cols,
        n_informative=_num_cols,
        n_redundant=0,
        weights=[0.75, 0.25],
        chunks=(_num_rows//10, _num_cols)
    )

    _X_ddf = ddf.from_dask_array(_X_da, columns=_column_names)
    _y_ddf = ddf.from_dask_array(_y_da)

    estimator = dask_LogisticRegression(
        penalty='l2',
        dual=False,
        tol=1e-6,
        C=1.0,
        fit_intercept=True,
        intercept_scaling=1,
        class_weight=None,
        random_state=None,
        solver='lbfgs',
        max_iter=10000,
        multi_class='auto',
        verbose=0,
        warm_start=False,
        n_jobs=-1
    )


    param_grid = [{'C': np.logspace(-3, 3, 7),'tol': np.logspace(-3, -1, 3)},
                  {'C': np.logspace(-7, -4, 4), 'tol': np.logspace(-6, -4, 3)}]


    @staticmethod
    def refit_1(X):
        # pizza, _GSTCV excepts on this
        _len = len(X['rank_test_balanced_accuracy'])
        return np.arange(_len)[X['rank_test_balanced_accuracy']==1][0]

    @staticmethod
    def refit_2(X):
        return 0

    @pytest.mark.parametrize('estimator', (estimator,))
    @pytest.mark.parametrize('param_grid', (param_grid,))
    @pytest.mark.parametrize('_refit', (refit_1, refit_2, 'balanced_accuracy'))
    @pytest.mark.parametrize('_X, _y', ((_X_da, _y_da), (_X_ddf, _y_ddf)))
    def test_dask_GSTCV(self, _X, _y, _refit, estimator, param_grid):

        TestCls = GSTCVDask(
            estimator,
            param_grid,
            scoring=['accuracy', 'balanced_accuracy'],
            thresholds=np.linspace(0,1,21),
            n_jobs=-1,
            cv=3,
            refit=_refit,
            error_score=np.nan,
            return_train_score=True,
            # OTHER POSSIBLE KWARGS FOR DASK SUPPORT
            iid=True,
            scheduler=None,
            cache_cv=False
        )

        TestCls.fit(_X, _y)


        DF = pd.DataFrame(TestCls.cv_results_)

        exp_rows = sum(map(np.prod, list(map(list, map(lambda x: map(len, x), map(dict.values, param_grid))))))

        assert DF.shape[0] == exp_rows

        # ATTRS ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
        ATTR_NAMES = [
            'cv_results_',
            'best_estimator_',
            'best_score_',
            'best_params_',
            'best_index_',
            'scorer_',
            'n_splits_'
        ]

        if callable(_refit):
            ATTR_NAMES.remove('best_score_')


        for attr in ATTR_NAMES:
            assert hasattr(TestCls, attr)

        # END ATTRS ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        # METHODS ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        NAMES = [
            'decision_function',
            'fit',
            'get_params',
            'inverse_transform',
            'predict',
            'predict_log_proba',
            'predict_proba',
            'score',
            'set_params',
            'transform',
            'visualize'
        ]

        for meth in NAMES:
            assert hasattr(TestCls, meth)
        # END METHODS ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
















