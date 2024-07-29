# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

import pytest

from model_selection.GSTCV._GSTCVDask._validation._dask_estimator import \
    _validate_dask_estimator

from sklearn.preprocessing import OneHotEncoder as sk_OneHotEncoder

from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import CountVectorizer as sk_CountVectorizer

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

from dask_ml.feature_extraction.text import CountVectorizer as dask_CountVectorizer

from dask_ml.preprocessing import OneHotEncoder as dask_OneHotEncoder


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


class TestValidateDaskEstimator:


    @pytest.mark.parametrize('not_instantiated',
        (sk_OneHotEncoder, sk_LinearRegression, sk_Ridge, sk_RidgeClassifier,
        sk_LogisticRegression, sk_SGDClassifier, sk_SGDRegressor,
        CalibratedClassifierCV, dask_OneHotEncoder, dask_LinearRegression,
        dask_LogisticRegression, XGBRegressor, XGBClassifier, XGBRanker,
        XGBRFRegressor, XGBRFClassifier, DaskXGBClassifier,
        DaskXGBRegressor, DaskXGBRanker, DaskXGBRFRegressor,
        DaskXGBRFClassifier, LGBMModel, LGBMClassifier, LGBMRegressor,
        LGBMRanker, DaskLGBMClassifier, DaskLGBMRegressor, DaskLGBMRanker)
    )
    def test_rejects_not_instantiated(self, not_instantiated):

        with pytest.raises(TypeError):
            _validate_dask_estimator(not_instantiated)


    @pytest.mark.parametrize('non_estimator',
        (int, str, list, object, sk_OneHotEncoder, dask_OneHotEncoder)
    )
    def test_rejects_non_estimator(self, non_estimator):

        with pytest.raises((AttributeError, TypeError)):
            _validate_dask_estimator(non_estimator())


        with pytest.raises(TypeError):
            _validate_dask_estimator(sk_CountVectorizer())

        with pytest.raises(TypeError):
            _validate_dask_estimator(sk_OneHotEncoder())

        with pytest.raises(AttributeError):
            _validate_dask_estimator(dask_CountVectorizer())

        with pytest.raises(AttributeError):
            _validate_dask_estimator(dask_OneHotEncoder())

    #########################





    @pytest.mark.parametrize('non_dask_classifier',
        (XGBClassifier, XGBRFClassifier, LGBMClassifier, sk_LogisticRegression)
    )
    def test_rejects_non_dask_classifiers(self, non_dask_classifier):

        with pytest.raises(TypeError):
            _validate_dask_estimator(non_dask_classifier())


    @pytest.mark.parametrize('non_classifier',
        (sk_LinearRegression, sk_Ridge, sk_SGDRegressor, XGBRanker, XGBRegressor,
         XGBRFRegressor, LGBMRegressor, LGBMRanker)
    )
    def test_rejects_non_dask_non_classifier(self, non_classifier):

        with pytest.raises(TypeError):
            _validate_dask_estimator(non_classifier())


    def test_rejects_non_dask_CCCV(self):

        with pytest.raises(TypeError):
            _validate_dask_estimator(CalibratedClassifierCV(sk_RidgeClassifier()))

        with pytest.raises(TypeError):
            _validate_dask_estimator(CalibratedClassifierCV(LGBMModel()))

        with pytest.raises(TypeError):
            _validate_dask_estimator(CalibratedClassifierCV(sk_SGDClassifier()))


    def test_accepts_dask_CCCV(self):
        _validate_dask_estimator(CalibratedClassifierCV(dask_LogisticRegression()))


    @pytest.mark.parametrize('dask_non_classifiers',
        (DaskXGBRegressor, DaskXGBRanker, DaskXGBRFRegressor,
        DaskLGBMRegressor, DaskLGBMRanker, dask_LinearRegression)
    )
    def test_rejects_all_dask_non_classifiers(self, dask_non_classifiers):

        # must be an instance not the class! & be a classifier!
        with pytest.raises(AttributeError):
            _validate_dask_estimator(dask_non_classifiers())




    @pytest.mark.parametrize('dask_classifiers',
        (DaskXGBClassifier, DaskXGBRFClassifier, DaskLGBMClassifier,
        dask_LogisticRegression)
    )
    def test_accepts_all_dask_classifiers(self, dask_classifiers):
        # must be an instance not the class! & be a classifier!
        _validate_dask_estimator(dask_classifiers())



    @pytest.mark.parametrize('junk_pipeline_steps',
        (
        [dask_OneHotEncoder(), dask_LogisticRegression()],
        [(4, dask_OneHotEncoder()), (3.14, dask_LogisticRegression())],
        [('onehot', 4), ('logistic', 3.14)]
        )
    )
    def test_rejects_pipeline_with_bad_steps(self, junk_pipeline_steps):
        # 24_07_27, unfortunately, sk pipeline does not do this, it will
        # allow bad steps (not in (str, cls()) format) and proceed and
        # return nonsensical results

        with pytest.raises(ValueError):
            _validate_dask_estimator(Pipeline(steps=junk_pipeline_steps))



    @pytest.mark.parametrize('good_pipeline_steps',
        ([('onehot', dask_OneHotEncoder()), ('logistic', dask_LogisticRegression())],)
    )
    def test_accepts_good_pipeline(self, good_pipeline_steps):

        _validate_dask_estimator(Pipeline(steps=good_pipeline_steps))





















