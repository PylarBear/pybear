# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

import pytest

from model_selection.GSTCV._GSTCVDask._validation._dask_estimator import \
    _validate_dask_estimator

from sklearn.preprocessing import OneHotEncoder as sk_OneHotEncoder

from sklearn.linear_model import (
    LinearRegression as sk_LinearRegression,
    Ridge as sk_Ridge,
    RidgeClassifier as sk_RidgeClassifier, # wrap with CCCV
    LogisticRegression as sk_LogisticRegression,
    SGDClassifier as sk_SGDClassifier,
    SGDRegressor as sk_SGDRegressor
)

from sklearn.calibration import CalibratedClassifierCV # wrap around RidgeClassifier

from sklearn.feature_extraction.text import CountVectorizer as sk_CountVectorizer
from sklearn.pipeline import Pipeline

# ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

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

class TestValidateWrappedDaskEstimator:

    # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **
    # CCCV ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **
    @pytest.mark.parametrize('non_dask_est',
        (sk_RidgeClassifier, LGBMModel, sk_SGDClassifier)
    )
    def test_warns_on_non_dask_CCCV(self, non_dask_est):

        exp_warn = (f"'{non_dask_est().__class__.__name__}' does not "
            f"appear to be a dask classifier.")
        with pytest.warns(match=exp_warn):
            _validate_dask_estimator(
                CalibratedClassifierCV(self._pipeline(non_dask_est()))
            )


    def test_accepts_dask_CCCV(self):

        _validate_dask_estimator(
            CalibratedClassifierCV(self._pipeline(dask_LogisticRegression()))
        )

    # END CCCV ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **
    # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

    # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **
    # pipeline - 2 inner objects ** * ** * ** * ** * ** * ** * ** * ** *

    def _pipeline(self, _estimator_instance):
        return Pipeline(
            steps=[
                      ('ct_vect', sk_CountVectorizer()),
                      ('clf', _estimator_instance)
            ]
        )


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


    def test_rejects_not_instantiated(self):

        with pytest.raises(TypeError):
            _validate_dask_estimator(Pipeline)


    @pytest.mark.parametrize('non_estimator', (int, str, list, object))
    def test_rejects_non_estimator(self, non_estimator):

        with pytest.raises(ValueError):
            _validate_dask_estimator(self._pipeline(non_estimator()))


    @pytest.mark.parametrize('non_classifier',
        (sk_LinearRegression, sk_Ridge, sk_SGDRegressor, XGBRanker,
         XGBRegressor, XGBRFRegressor, LGBMRegressor, LGBMRanker)
    )
    def test_rejects_non_dask_non_classifier(self, non_classifier):

        exp_warn = (f"'{non_classifier().__class__.__name__}' does not "
            f"appear to be a dask classifier.")
        with pytest.warns(match=exp_warn):
            with pytest.raises(AttributeError):
                _validate_dask_estimator(self._pipeline(non_classifier()))

        # the old way pre-warn
        # with pytest.raises(TypeError):
        #     _validate_dask_estimator(self._pipeline(non_classifier()))


    @pytest.mark.parametrize('non_dask_classifier',
        (XGBClassifier, XGBRFClassifier, LGBMClassifier, sk_LogisticRegression)
    )
    def test_warns_on_non_dask_classifiers(self, non_dask_classifier):

        exp_warn = (f"'{non_dask_classifier().__class__.__name__}' does not "
            f"appear to be a dask classifier.")
        with pytest.warns(match=exp_warn):
            _validate_dask_estimator(self._pipeline(non_dask_classifier()))

        # the old way pre-warn
        # with pytest.raises(TypeError):
        #     _validate_dask_estimator(self._pipeline(non_dask_classifier()))


    @pytest.mark.parametrize('dask_non_classifiers',
        (DaskXGBRegressor, DaskXGBRanker, DaskXGBRFRegressor, DaskLGBMRegressor,
         DaskLGBMRanker, dask_LinearRegression, dask_OneHotEncoder)
    )
    def test_rejects_all_dask_non_classifiers(self, dask_non_classifiers):

        # must be an instance not the class! & be a classifier!
        with pytest.raises(AttributeError):
            _validate_dask_estimator(self._pipeline(dask_non_classifiers()))


    @pytest.mark.parametrize('dask_classifiers',
        (DaskXGBClassifier, DaskXGBRFClassifier, DaskLGBMClassifier,
        dask_LogisticRegression)
    )
    def test_accepts_good_pipeline_1(self, dask_classifiers):
        # must be an instance not the class! & be a classifier!
        _validate_dask_estimator(self._pipeline(dask_classifiers()))


    @pytest.mark.parametrize('good_pipeline_steps',
        ([('onehot', dask_OneHotEncoder()), ('logistic', dask_LogisticRegression())],)
    )
    def test_accepts_good_pipeline_2(self, good_pipeline_steps):

        _validate_dask_estimator(Pipeline(steps=good_pipeline_steps))


    def test_warns_on_wrapped_non_dask_CCCV(self):

        exp_warn = (f"'{sk_RidgeClassifier().__class__.__name__}' does not "
            f"appear to be a dask classifier.")
        with pytest.warns(match=exp_warn):
            _validate_dask_estimator(
                self._pipeline(CalibratedClassifierCV(sk_RidgeClassifier()))
            )

        exp_warn = (f"'{LGBMModel().__class__.__name__}' does not "
            f"appear to be a dask classifier.")
        with pytest.warns(match=exp_warn):
            _validate_dask_estimator(
                self._pipeline(CalibratedClassifierCV(LGBMModel()))
            )

        exp_warn = (f"'{sk_SGDClassifier().__class__.__name__}' does not "
            f"appear to be a dask classifier.")
        with pytest.warns(match=exp_warn):
            _validate_dask_estimator(
                self._pipeline(CalibratedClassifierCV(sk_SGDClassifier()))
            )


    def test_accepts_wrapped_dask_CCCV(self):
        _validate_dask_estimator(
            self._pipeline(CalibratedClassifierCV(dask_LogisticRegression()))
        )


    # END pipeline - 2 inner objects ** * ** * ** * ** * ** * ** * ** *
    # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **


    # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **
    # pipeline - 3 inner objects ** * ** * ** * ** * ** * ** * ** * ** *

    @staticmethod
    @pytest.fixture
    def sk_pipeline():
        # must be an instance not the class!
        return Pipeline(
            steps=[
                ('sk_CountVectorizer', sk_CountVectorizer()),
                ('sk_OneHotEncoder', sk_OneHotEncoder()),
                ('sk_Logistic', sk_LogisticRegression())
            ],
            verbose=0
        )


    @staticmethod
    @pytest.fixture
    def dask_pipeline_1():
        # must be an instance not the class!
        return Pipeline(
            steps=[
                ('dask_CountVectorizer', dask_CountVectorizer()),
                ('dask_OneHotEncoder', dask_OneHotEncoder()),
                ('dask_Logistic', dask_LogisticRegression())
            ],
            verbose=0
        )


    @staticmethod
    @pytest.fixture
    def dask_pipeline_2():
        # must be an instance not the class!
        return Pipeline(
            steps=[
                ('sk_CountVectorizer', sk_CountVectorizer()),
                ('sk_OneHotEncoder', sk_OneHotEncoder()),
                ('dask_Logistic', dask_LogisticRegression())
            ],
            verbose=0
        )


    @staticmethod
    @pytest.fixture
    def dask_pipeline_3():
        # must be an instance not the class!
        return Pipeline(
            steps=[
                ('dask_CountVectorizer', dask_CountVectorizer()),
                ('dask_OneHotEncoder', dask_OneHotEncoder()),
                ('sk_Logistic', sk_LogisticRegression())
            ],
            verbose=0
        )


    def test_accuracy_pipeline(self,
        sk_pipeline, dask_pipeline_1, dask_pipeline_2, dask_pipeline_3
        ):

        exp_warn = (f"'{sk_LogisticRegression().__class__.__name__}' does not "
            f"appear to be a dask classifier.")
        with pytest.warns(match=exp_warn):
            _validate_dask_estimator(sk_pipeline)

        # the old way pre-warn
        # with pytest.raises(TypeError):
        #     _validate_dask_estimator(sk_pipeline)

        _validate_dask_estimator(dask_pipeline_1)
        _validate_dask_estimator(dask_pipeline_2)

        exp_warn = (f"'{sk_LogisticRegression().__class__.__name__}' does not "
            f"appear to be a dask classifier.")
        with pytest.warns(match=exp_warn):
            _validate_dask_estimator(dask_pipeline_3)

        # the old way pre-warn
        # with pytest.raises(TypeError):
        #     _validate_dask_estimator(dask_pipeline_3)


    # END pipeline - 3 inner objects ** * ** * ** * ** * ** * ** * ** *
    # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **













