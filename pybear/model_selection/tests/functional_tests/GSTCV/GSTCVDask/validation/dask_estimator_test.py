# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

import pytest

from model_selection.GSTCV._GSTCVDask._validation._dask_estimator import \
    _validate_dask_estimator

from sklearn.preprocessing import OneHotEncoder as sk_OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer as sk_CountVectorizer

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

        with pytest.raises(
            TypeError,
            match=f"estimator must be an instance, not the class"
        ):
            _validate_dask_estimator(not_instantiated)


    @pytest.mark.parametrize('non_estimator',
        (int, str, list, object, sk_OneHotEncoder, dask_OneHotEncoder,
         sk_CountVectorizer, dask_CountVectorizer)
    )
    def test_rejects_non_estimator(self, non_estimator):

        with pytest.raises(AttributeError):
            _validate_dask_estimator(non_estimator())


    @pytest.mark.parametrize('non_dask_classifier',
        (XGBClassifier, XGBRFClassifier, LGBMClassifier, sk_LogisticRegression)
    )
    def test_warns_on_non_dask_classifiers(self, non_dask_classifier):

        exp_warn = (f"'{non_dask_classifier().__class__.__name__}' does not "
            f"appear to be a dask classifier.")
        with pytest.warns(match=exp_warn):
            _validate_dask_estimator(non_dask_classifier())


        # the old way pre-warn
        # with pytest.raises(TypeError):
        #     _validate_dask_estimator(non_dask_classifier())


    @pytest.mark.parametrize('non_classifier',
        (sk_LinearRegression, sk_Ridge, sk_SGDRegressor, XGBRanker, XGBRegressor,
         XGBRFRegressor, LGBMRegressor, LGBMRanker)
    )
    def test_rejects_non_dask_non_classifier_with_warn(self, non_classifier):

        exp_warn = (f"'{non_classifier().__class__.__name__}' does not "
            f"appear to be a dask classifier.")
        with pytest.warns(match=exp_warn):
            with pytest.raises(AttributeError):
                _validate_dask_estimator(non_classifier())

        # the old way pre-warn
        # with pytest.raises(TypeError):
        #     _validate_dask_estimator(non_classifier())


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















