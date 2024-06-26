# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pytest

from model_selection.GSTCV._validation._is_dask_estimator \
    import _is_dask_estimator


# sklearn
from sklearn.pipeline import Pipeline
from sklearn.linear_model import (
    LogisticRegression as sk_LogisticRegression,
    LinearRegression as sk_LinearRegression,
    RidgeClassifier as sk_RidgeClassifier
)
from sklearn.feature_extraction.text import (
    CountVectorizer as sk_CountVectorizer
)
from sklearn.preprocessing import (
    OneHotEncoder as sk_OneHotEncoder
)

# dask
from dask_ml.linear_model import (
    LogisticRegression as dask_LogisticRegression,
    LinearRegression as dask_LinearRegression
)
from dask_ml.feature_extraction.text import (
    CountVectorizer as dask_CountVectorizer
)
from dask_ml.preprocessing import(
    OneHotEncoder as dask_OneHotEncoder
)



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





class TestIsDaskEstimator:


    @pytest.mark.parametrize('non_sk_dsk_classes',
        (str, list, int, object)
    )
    def test_rejects_non_sk_dsk_classes(self, non_sk_dsk_classes):

        with pytest.raises(AttributeError):
            # must be an instance not the class!
            _is_dask_estimator(non_sk_dsk_classes())


    def test_reject_non_estimator(self):

        with pytest.raises(AttributeError):
            _is_dask_estimator(sk_CountVectorizer())

        with pytest.raises(AttributeError):
            _is_dask_estimator(sk_OneHotEncoder())

        with pytest.raises(AttributeError):
            _is_dask_estimator(dask_CountVectorizer())

        with pytest.raises(AttributeError):
            _is_dask_estimator(dask_OneHotEncoder())



    def test_accuracy_pipeline(self,
        sk_pipeline, dask_pipeline_1, dask_pipeline_2, dask_pipeline_3
        ):

        assert not _is_dask_estimator(sk_pipeline)
        assert _is_dask_estimator(dask_pipeline_1)
        assert _is_dask_estimator(dask_pipeline_2)
        assert not _is_dask_estimator(dask_pipeline_3)


    def test_accuracy_estimator(self):

        # must be an instance not the class! & be an estimator!
        assert not _is_dask_estimator(sk_RidgeClassifier())
        assert not _is_dask_estimator(sk_LogisticRegression())
        assert not _is_dask_estimator(sk_LinearRegression())
        assert _is_dask_estimator(dask_LogisticRegression())
        assert _is_dask_estimator(dask_LinearRegression())













