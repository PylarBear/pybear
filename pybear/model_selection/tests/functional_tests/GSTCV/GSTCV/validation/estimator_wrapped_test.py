# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

import pytest

from model_selection.GSTCV._GSTCV._validation._estimator import \
    _validate_estimator

from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import (
    LinearRegression as sk_LinearRegression,
    Ridge as sk_Ridge,
    RidgeClassifier as sk_RidgeClassifier, # wrap with CCCV
    LogisticRegression as sk_LogisticRegression,
    SGDClassifier as sk_SGDClassifier,
    SGDRegressor as sk_SGDRegressor
)

from sklearn.calibration import CalibratedClassifierCV # wrap around RidgeClassifier

from dask_ml.linear_model import (
    LinearRegression as dask_LinearRegression,
    LogisticRegression as dask_LogisticRegression
)

from xgboost import (
    XGBRegressor,
    XGBClassifier,
    XGBRanker,
    XGBRFRegressor,
    XGBRFClassifier
)

from xgboost.dask import (
    DaskXGBClassifier,
    DaskXGBRegressor,
    DaskXGBRanker,
    DaskXGBRFRegressor,
    DaskXGBRFClassifier
)

from lightgbm import (
    LGBMModel,
    LGBMClassifier,
    LGBMRegressor,
    LGBMRanker
)

from lightgbm import (
    DaskLGBMClassifier,
    DaskLGBMRegressor,
    DaskLGBMRanker
)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline





# must be an instance not the class! & be an estimator!


class TestValidateWrappedEstimator:


    def _pipeline(self, _estimator_instance):
        return Pipeline(
            steps=[
                      ('ct_vect', CountVectorizer()),
                      ('clf', _estimator_instance)
            ]
        )


    def test_rejects_not_instantiated(self):

        with pytest.raises(TypeError):
            _validate_estimator(Pipeline)


    @pytest.mark.parametrize('non_estimator',
        (int, str, list, object, OneHotEncoder)
    )
    def test_rejects_non_estimator(self, non_estimator):

        with pytest.raises(AttributeError):
            _validate_estimator(self._pipeline(non_estimator()))


    @pytest.mark.parametrize('non_classifier',
        (sk_LinearRegression, sk_Ridge, sk_SGDRegressor, XGBRanker, XGBRegressor,
         XGBRFRegressor, LGBMRegressor, LGBMRanker)
    )
    def test_rejects_non_classifier(self, non_classifier):

        with pytest.raises(AttributeError):
            _validate_estimator(self._pipeline(non_classifier()))


    @pytest.mark.parametrize('good_classifiers',
        (XGBClassifier, XGBRFClassifier, LGBMClassifier, sk_LogisticRegression)
    )
    def test_accepts_non_dask_classifiers(self, good_classifiers):
        _validate_estimator(self._pipeline(good_classifiers()))


    def test_accepts_non_dask_CCCV(self):
        _validate_estimator(CalibratedClassifierCV(self._pipeline(sk_RidgeClassifier())))
        _validate_estimator(CalibratedClassifierCV(self._pipeline(LGBMModel())))
        _validate_estimator(CalibratedClassifierCV(self._pipeline(sk_SGDClassifier())))


    def test_rejects_dask_CCCV(self):
        with pytest.raises(TypeError):
            _validate_estimator(
                CalibratedClassifierCV(self._pipeline(dask_LogisticRegression()))
            )


    @pytest.mark.parametrize('dask_non_classifiers',
        (DaskXGBRegressor, DaskXGBRanker, DaskXGBRFRegressor,
        DaskLGBMRegressor, DaskLGBMRanker, dask_LinearRegression)
    )
    def test_rejects_all_dask_non_classifiers(self, dask_non_classifiers):

        # must be an instance not the class! & be a classifier!
        with pytest.raises(TypeError):
            _validate_estimator(self._pipeline(dask_non_classifiers()))


    @pytest.mark.parametrize('dask_classifiers',
        (DaskXGBClassifier, DaskXGBRFClassifier, DaskLGBMClassifier,
        dask_LogisticRegression)
    )
    def test_rejects_all_dask_classifiers(self, dask_classifiers):
        # must be an instance not the class! & be a classifier!
        with pytest.raises(TypeError):
            _validate_estimator(self._pipeline(dask_classifiers()))

























