# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pytest

import numpy as np


from sklearn.linear_model import LogisticRegression as sk_LogisticRegression
from sklearn.preprocessing import OneHotEncoder as sk_OneHotEncoder
from sklearn.model_selection import GridSearchCV as sk_GridSearchCV

from dask_ml.linear_model import LogisticRegression as dask_LogisticRegression
from dask_ml.preprocessing import OneHotEncoder as dask_OneHotEncoder
from dask_ml.model_selection import GridSearchCV as dask_GridSearchCV

from model_selection.GSTCV._GSTCV.GSTCV import GSTCV
from model_selection.GSTCV._GSTCVDask.GSTCVDask import GSTCVDask

from sklearn.pipeline import Pipeline


# pre-fit, all attrs should not be available and should except. how they
# except differs for sklearn and dask


class TestAttrsPreFit:

    @staticmethod
    def param_grid():
        return {'C': np.logspace(-5, -2, 4)}


    # sklearn ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

    @staticmethod
    def sk_est():
        return sk_LogisticRegression(C=1e-3)


    @staticmethod
    def sk_GSCV(refit):
        return sk_GridSearchCV(
            estimator=sk_est,
            param_grid=param_grid,
            refit=refit
        )

    @staticmethod
    def sk_GSTCV(refit):
        return GSTCV(
            estimator=sk_est,
            param_grid=param_grid,
            refit=refit
        )

    # END sklearn ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    # dask ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    @staticmethod
    def dask_est():
        return dask_LogisticRegression(C=1e-3)


    @staticmethod
    def dask_GSCV(refit):
        return dask_GridSearchCV(
            estimator=dask_est,
            param_grid=param_grid,
            refit=refit
        )

    @staticmethod
    def dask_GSTCV(refit):
        return GSTCV(
            estimator=dask_est,
            param_grid=param_grid,
            refit=refit
        )
    # END dask ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **



    @pytest.mark.parametrize('sk_GSCV, sk_GSTCV, dask_GSCV, dask_GSTCV',
         ((sk_GSCV, sk_GSTCV, dask_GSCV, dask_GSTCV),)
    )
    @pytest.mark.parametrize('refit', (False, 'balanced_accuracy', lambda x: 0))
    def test_cv_results(self, sk_GSCV, sk_GSTCV, dask_GSCV, dask_GSTCV, refit):
        with pytest.raises(AttributeError):
            sk_GSCV(refit=refit).cv_results_

        with pytest.raises(AttributeError):
            sk_GSTCV(refit=refit).cv_results_

        with pytest.raises(AttributeError):
            dask_GSCV(refit=refit).cv_results_

        with pytest.raises(AttributeError):
            dask_GSTCV(refit=refit).cv_results_


    @pytest.mark.parametrize('refit', (False, 'balanced_accuracy', lambda x: 0))
    def test_best_estimator(self, sk_GSCV, sk_GSTCV, dask_GSCV, dask_GSTCV, refit):
        with pytest.raises(AttributeError):
            sk_GSCV(refit=refit).best_estimator_

        with pytest.raises(AttributeError):
            sk_GSTCV(refit=refit).best_estimator_

        with pytest.raises(AttributeError):
            dask_GSCV(refit=refit).best_estimator_

        with pytest.raises(AttributeError):
            dask_GSTCV(refit=refit).best_estimator_



    @pytest.mark.parametrize('refit',(False, 'balanced_accuracy', lambda x: 0))
    def test_best_score(self, sk_GSCV, sk_GSTCV, dask_GSCV, dask_GSTCV, refit):
        with pytest.raises(AttributeError):
            sk_GSCV(refit=refit).best_score_

        with pytest.raises(AttributeError):
            sk_GSTCV(refit=refit).best_score_

        with pytest.raises(AttributeError):
            dask_GSCV(refit=refit).best_score_

        with pytest.raises(AttributeError):
            dask_GSTCV(refit=refit).best_score_


    @pytest.mark.parametrize('refit', (False, 'balanced_accuracy', lambda x: 0))
    def test_best_params(self, sk_GSCV, sk_GSTCV, dask_GSCV, dask_GSTCV, refit):
        with pytest.raises(AttributeError):
            sk_GSCV(refit=refit).best_params_

        with pytest.raises(AttributeError):
            sk_GSTCV(refit=refit).best_params_

        with pytest.raises(AttributeError):
            dask_GSCV(refit=refit).best_params_

        with pytest.raises(AttributeError):
            dask_GSTCV(refit=refit).best_params_


    @pytest.mark.parametrize('refit', (False, 'balanced_accuracy', lambda x: 0))
    def test_best_index(self, sk_GSCV, sk_GSTCV, dask_GSCV, dask_GSTCV, refit):
        with pytest.raises(AttributeError):
            sk_GSCV(refit=refit).best_index_

        with pytest.raises(AttributeError):
            sk_GSTCV(refit=refit).best_index_

        with pytest.raises(AttributeError):
            dask_GSCV(refit=refit).best_index_

        with pytest.raises(AttributeError):
            dask_GSTCV(refit=refit).best_index_


    @pytest.mark.parametrize('refit', (False, 'balanced_accuracy', lambda x: 0))
    def test_scorer(self, sk_GSCV, sk_GSTCV, dask_GSCV, dask_GSTCV, refit):
        with pytest.raises(AttributeError):
            sk_GSCV(refit=refit).scorer_

        with pytest.raises(AttributeError):
            sk_GSTCV(refit=refit).scorer_

        with pytest.raises(AttributeError):
            dask_GSCV(refit=refit).scorer_

        with pytest.raises(AttributeError):
            dask_GSTCV(refit=refit).scorer_


    @pytest.mark.parametrize('refit', (False, 'balanced_accuracy', lambda x: 0))
    def test_n_splits(self, sk_GSCV, sk_GSTCV, dask_GSCV, dask_GSTCV, refit):
        with pytest.raises(AttributeError):
            sk_GSCV(refit=refit).n_splits_

        with pytest.raises(AttributeError):
            sk_GSTCV(refit=refit).n_splits_

        with pytest.raises(AttributeError):
            dask_GSCV(refit=refit).n_splits_

        with pytest.raises(AttributeError):
            dask_GSTCV(refit=refit).n_splits_


    @pytest.mark.parametrize('refit', (False, 'balanced_accuracy', lambda x: 0))
    def test_refit_time(self, sk_GSCV, sk_GSTCV, dask_GSCV, dask_GSTCV, refit):
        with pytest.raises(AttributeError):
            sk_GSCV(refit=refit).refit_time_

        with pytest.raises(AttributeError):
            sk_GSTCV(refit=refit).refit_time_

        with pytest.raises(AttributeError):
            dask_GSCV(refit=refit).refit_time_

        with pytest.raises(AttributeError):
            dask_GSTCV(refit=refit).refit_time_


    @pytest.mark.parametrize('refit', (False, 'balanced_accuracy', lambda x: 0))
    def test_multimetric(self, sk_GSCV, sk_GSTCV, dask_GSCV, dask_GSTCV, refit):
        with pytest.raises(AttributeError):
            sk_GSCV(refit=refit).multimetric_

        with pytest.raises(AttributeError):
            sk_GSTCV(refit=refit).multimetric_

        with pytest.raises(AttributeError):
            dask_GSCV(refit=refit).multimetric_

        with pytest.raises(AttributeError):
            dask_GSTCV(refit=refit).multimetric_


    @pytest.mark.parametrize('refit', (False, 'balanced_accuracy', lambda x: 0))
    def test_classes(self, sk_GSCV, sk_GSTCV, dask_GSCV, dask_GSTCV, refit):
        with pytest.raises(AttributeError):
            sk_GSCV(refit=refit).classes_

        with pytest.raises(AttributeError):
            sk_GSTCV(refit=refit).classes_

        with pytest.raises(AttributeError):
            dask_GSCV(refit=refit).classes_

        with pytest.raises(AttributeError):
            dask_GSTCV(refit=refit).classes_


    @pytest.mark.parametrize('refit', (False, 'balanced_accuracy', lambda x: 0))
    def test_n_features_in(self, sk_GSCV, sk_GSTCV, dask_GSCV, dask_GSTCV, refit):
        with pytest.raises(AttributeError):
            sk_GSCV(refit=refit).n_features_in_

        with pytest.raises(AttributeError):
            sk_GSTCV(refit=refit).n_features_in_

        with pytest.raises(AttributeError):
            dask_GSCV(refit=refit).n_features_in_

        with pytest.raises(AttributeError):
            dask_GSTCV(refit=refit).n_features_in_


    @pytest.mark.parametrize('refit', (False, 'balanced_accuracy', lambda x: 0))
    def test_feature_names_in(self, sk_GSCV, sk_GSTCV, dask_GSCV, dask_GSTCV, refit):

        with pytest.raises(AttributeError):
            sk_GSCV(refit=refit).feature_names_in_

        with pytest.raises(AttributeError):
            sk_GSTCV(refit=refit).feature_names_in_

        with pytest.raises(AttributeError):
            dask_GSCV(refit=refit).feature_names_in_

        with pytest.raises(AttributeError):
            dask_GSTCV(refit=refit).feature_names_in_












































