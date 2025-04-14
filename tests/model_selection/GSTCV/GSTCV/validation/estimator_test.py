# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

from pybear.model_selection.GSTCV._GSTCV._validation._estimator import \
    _validate_estimator

from sklearn.preprocessing import OneHotEncoder as sk_OneHotEncoder

# wrap around RidgeClassifier
from sklearn.calibration import CalibratedClassifierCV

from sklearn.linear_model import (
    LinearRegression as sk_LinearRegression,
    Ridge as sk_Ridge,
    RidgeClassifier as sk_RidgeClassifier, # wrap with CCCV
    LogisticRegression as sk_LogisticRegression,
    SGDClassifier as sk_SGDClassifier,
    SGDRegressor as sk_SGDRegressor
)

# ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

from dask_ml.linear_model import (
    LinearRegression as dask_LinearRegression,
    LogisticRegression as dask_LogisticRegression
)


# must be an instance not the class! & be an estimator!



class TestValidateEstimator:


    @pytest.mark.parametrize('not_instantiated',
        (sk_OneHotEncoder, sk_LinearRegression, sk_Ridge, sk_RidgeClassifier,
        sk_LogisticRegression, sk_SGDClassifier, sk_SGDRegressor,
        CalibratedClassifierCV)
    )
    def test_rejects_not_instantiated(self, not_instantiated):

        with pytest.raises(
            TypeError,
            match=f"estimator must be an instance, not the class"
        ):
            _validate_estimator(not_instantiated)


    @pytest.mark.parametrize('non_estimator',
        (int, str, list, object, sk_OneHotEncoder)
    )
    def test_rejects_non_estimator(self, non_estimator):

        with pytest.raises(AttributeError):
            _validate_estimator(non_estimator())


    @pytest.mark.parametrize('non_classifier',
        (sk_LinearRegression, sk_Ridge, sk_SGDRegressor)
    )
    def test_rejects_non_classifier(self, non_classifier):

        with pytest.raises(AttributeError):
            _validate_estimator(non_classifier())


    @pytest.mark.parametrize('good_classifiers', (sk_LogisticRegression, ))
    def test_accepts_non_dask_classifiers(self, good_classifiers):
        _validate_estimator(good_classifiers())


    @pytest.mark.parametrize('dask_non_classifiers', (dask_LinearRegression, ))
    def test_rejects_all_dask_non_classifiers(self, dask_non_classifiers):

        # must be an instance not the class! & be a classifier!
        with pytest.raises(TypeError):
            _validate_estimator(dask_non_classifiers())


    @pytest.mark.parametrize('dask_classifiers', (dask_LogisticRegression, ))
    def test_rejects_all_dask_classifiers(self, dask_classifiers):
        # must be an instance not the class! & be a classifier!
        with pytest.raises(TypeError):
            _validate_estimator(dask_classifiers())














