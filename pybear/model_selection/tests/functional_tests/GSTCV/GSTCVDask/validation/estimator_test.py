# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

import pytest

from model_selection.GSTCV._GSTCVDask._validation._estimator import _val_estimator

from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import (
    LinearRegression as sk_LinearRegression,
    RidgeClassifier as sk_RidgeClassifier,
    LogisticRegression as sk_LogisticRegression,



)
from dask_ml.linear_model import (
    LinearRegression as dask_LinearRegression,
    LogisticRegression as dask_LogisticRegression
)


# must be an instance not the class! & be an estimator!


class TestValEstimator:



    @pytest.mark.parametrize('non_estimator',
        (int, str, list, object, OneHotEncoder)
    )
    def test_rejects_non_estimator(self, non_estimator):

        with pytest.raises(AttributeError):
            _val_estimator(non_estimator())


    @pytest.mark.parametrize('non_classifier',
        (sk_LinearRegression, dask_LinearRegression)
    )
    def test_rejects_non_classifier(self, non_classifier):

        with pytest.raises(TypeError):
            _val_estimator(non_classifier())


    def test_dask_estimator_accuracy(self):

        # must be an instance not the class! & be a classifier!
        assert not _val_estimator(sk_RidgeClassifier())
        assert not _val_estimator(sk_LogisticRegression())
        assert _val_estimator(dask_LogisticRegression())
























