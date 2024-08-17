# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pytest
import numpy as np
from dask_ml.linear_model import LogisticRegression as dask_LogisticRegression
from model_selection.GSTCV._GSTCVDask.GSTCVDask import GSTCVDask


# pre-fit, all attrs should not be available and should except.

class TestAttrsPreFit:

    @staticmethod
    @pytest.fixture
    def param_grid():
        return {'C': np.logspace(-5, -2, 4)}


    @staticmethod
    @pytest.fixture
    def dask_est():
        return dask_LogisticRegression(C=1e-3)


    @staticmethod
    @pytest.fixture
    def dask_GSTCV(dask_est, param_grid):

        def foo(refit=False, scoring=None):

            return GSTCVDask(
                estimator=dask_est,
                param_grid=param_grid,
                scoring=scoring,
                refit=refit
            )

        return foo


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

    # end fixtures #####################################################

    @pytest.mark.parametrize('attr',
        ('cv_results_', 'best_estimator_', 'best_index_', 'scorer_', 'n_splits_',
         'refit_time_', 'multimetric_', 'feature_names_in_', 'best_threshold_')
    )
    @pytest.mark.parametrize('_scoring',
        ('balanced_accuracy', ['accuracy', 'balanced_accuracy'])
    )
    @pytest.mark.parametrize('refit', (False, 'balanced_accuracy', lambda x: 0))
    def test_attrs_1(self, refit, dask_GSTCV, generic_no_attribute, attr, _scoring):

        with pytest.raises(
            AttributeError,
            match=generic_no_attribute('GSTCVDask', attr)
        ):
            getattr(dask_GSTCV(refit=refit, scoring=_scoring), attr)


    @pytest.mark.parametrize('attr', ('best_score_', 'best_params_'))
    @pytest.mark.parametrize('_scoring',
        ('balanced_accuracy', ['accuracy', 'balanced_accuracy'])
    )
    @pytest.mark.parametrize('refit',(False, 'balanced_accuracy', lambda x: 0))
    def test_attrs_2(self, dask_GSTCV, refit, generic_no_attribute, generic_not_fitted,
        attr, _scoring
    ):

        with pytest.raises(
            AttributeError,
            match=generic_no_attribute('GSTCVDask', attr)
        ):
            getattr(dask_GSTCV(refit=refit, scoring=_scoring), attr)


    @pytest.mark.parametrize('_scoring',
        ('balanced_accuracy', ['accuracy', 'balanced_accuracy'])
    )
    @pytest.mark.parametrize('refit', (False, 'balanced_accuracy', lambda x: 0))
    def test_classes(self, dask_GSTCV, refit, generic_no_attribute, _scoring):

        if refit is False:

            exc_info = lambda x: (f"This {x} instance was initialized with "
                f"`refit=False`. classes_ is available only after refitting on "
                f"the best parameters. You can refit an estimator manually using "
                f"the `best_params_` attribute")

            with pytest.raises(AttributeError, match=exc_info('GSTCVDask')):
                dask_GSTCV(refit=refit, scoring=_scoring).classes_

        elif refit == 'balanced_accuracy' or callable(refit):

            exc_info = lambda x: (f"This {x} instance is not fitted yet. Call "
                f"'fit' with appropriate arguments before using this estimator.")

            with pytest.raises(AttributeError, match=exc_info('GSTCVDask')):
                dask_GSTCV(refit=refit, scoring=_scoring).classes_

        else:
            raise Exception(f"unexpected refit '{refit}'")


    @pytest.mark.parametrize('_scoring',
        ('balanced_accuracy', ['accuracy', 'balanced_accuracy'])
    )
    @pytest.mark.parametrize('refit', (False, 'balanced_accuracy', lambda x: 0))
    def test_n_features_in(self, dask_GSTCV, refit, generic_no_attribute, _scoring):

        exc_info = lambda x: f'{x} object has no n_features_in_ attribute.'

        with pytest.raises(AttributeError, match=exc_info("GSTCVDask")):
            dask_GSTCV(refit=refit, scoring=_scoring).n_features_in_














