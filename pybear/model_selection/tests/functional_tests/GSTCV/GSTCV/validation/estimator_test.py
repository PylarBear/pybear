# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

import pytest

from model_selection.GSTCV._GSTCV._validation._estimator import \
    _validate_estimator

from sklearn.preprocessing import OneHotEncoder as sk_OneHotEncoder

from sklearn.pipeline import Pipeline

from sklearn.calibration import CalibratedClassifierCV # wrap around RidgeClassifier

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








# must be an instance not the class! & be an estimator!


class TestValidateEstimator:


    @pytest.mark.parametrize('not_instantiated',
        (sk_OneHotEncoder, sk_LinearRegression, sk_Ridge, sk_RidgeClassifier,
        sk_LogisticRegression, sk_SGDClassifier, sk_SGDRegressor,
        CalibratedClassifierCV, dask_LinearRegression,
        dask_LogisticRegression, XGBRegressor, XGBClassifier,
        XGBRanker, XGBRFRegressor, XGBRFClassifier, DaskXGBClassifier,
        DaskXGBRegressor, DaskXGBRanker, DaskXGBRFRegressor,
        DaskXGBRFClassifier, LGBMModel, LGBMClassifier, LGBMRegressor,
        LGBMRanker, DaskLGBMClassifier, DaskLGBMRegressor, DaskLGBMRanker)
    )
    def test_rejects_not_instantiated(self, not_instantiated):

        with pytest.raises(TypeError):
            _validate_estimator(not_instantiated)


    @pytest.mark.parametrize('non_estimator',
        (int, str, list, object, sk_OneHotEncoder)
    )
    def test_rejects_non_estimator(self, non_estimator):

        with pytest.raises(AttributeError):
            _validate_estimator(non_estimator())


    @pytest.mark.parametrize('non_classifier',
        (sk_LinearRegression, sk_Ridge, sk_SGDRegressor, XGBRanker, XGBRegressor,
         XGBRFRegressor, LGBMRegressor, LGBMRanker)
    )
    def test_rejects_non_classifier(self, non_classifier):

        with pytest.raises(AttributeError):
            _validate_estimator(non_classifier())


    @pytest.mark.parametrize('good_classifiers',
        (XGBClassifier, XGBRFClassifier, LGBMClassifier, sk_LogisticRegression)
    )
    def test_accepts_non_dask_classifiers(self, good_classifiers):
        _validate_estimator(good_classifiers())


    def test_accepts_non_dask_CCCV(self):
        _validate_estimator(CalibratedClassifierCV(sk_RidgeClassifier()))
        _validate_estimator(CalibratedClassifierCV(LGBMModel()))
        _validate_estimator(CalibratedClassifierCV(sk_SGDClassifier()))


    def test_rejects_dask_CCCV(self):
        with pytest.raises(TypeError):
            _validate_estimator(CalibratedClassifierCV(dask_LogisticRegression()))


    @pytest.mark.parametrize('dask_non_classifiers',
        (DaskXGBRegressor, DaskXGBRanker, DaskXGBRFRegressor,
        DaskLGBMRegressor, DaskLGBMRanker, dask_LinearRegression)
    )
    def test_rejects_all_dask_non_classifiers(self, dask_non_classifiers):

        # must be an instance not the class! & be a classifier!
        with pytest.raises(TypeError):
            _validate_estimator(dask_non_classifiers())


    @pytest.mark.parametrize('dask_classifiers',
        (DaskXGBClassifier, DaskXGBRFClassifier, DaskLGBMClassifier,
        dask_LogisticRegression)
    )
    def test_rejects_all_dask_classifiers(self, dask_classifiers):
        # must be an instance not the class! & be a classifier!
        with pytest.raises(TypeError):
            _validate_estimator(dask_classifiers())



    @pytest.mark.parametrize('junk_pipeline_steps',
        (
        [sk_OneHotEncoder(), sk_LogisticRegression()],
        [(4, sk_OneHotEncoder()), (3.14, sk_LogisticRegression())],
        [('onehot', 4), ('logistic', 3.14)]
        )
    )
    def test_rejects_pipeline_with_bad_steps(self, junk_pipeline_steps):
        # 24_07_27, unfortunately, sk pipeline does not do this, it will
        # allow bad steps (not in (str, cls()) format) and proceed and
        # return nonsensical results

        with pytest.raises(ValueError):
            _validate_estimator(Pipeline(steps=junk_pipeline_steps))



    @pytest.mark.parametrize('good_pipeline_steps',
        ([('onehot', sk_OneHotEncoder()), ('logistic', sk_LogisticRegression())],)
    )
    def test_accepts_good_pipeline(self, good_pipeline_steps):

        _validate_estimator(Pipeline(steps=good_pipeline_steps))


















