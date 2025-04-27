# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

from distributed import Client
from pybear.model_selection.GSTCV._GSTCVDask.GSTCVDask import GSTCVDask

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



class TestGSTCVInput:

    # only test args/kwargs unique to GSTCVDask that are not already covered
    # by GSTCV testing, because both share the mixin.
    # (estimator, cache_cv, iid, scheduler)


    # def __init__(self,
    #     estimator,
    #     param_grid: ParamGridType,
    #     *,
    #     # thresholds can be a single number or list-type passed in
    #     # param_grid or applied universally via thresholds kwarg
    #     thresholds:
    #           Optional[Union[Iterable[Union[int, float]], int, float, None]]=None,
    #     scoring: Optional[ScorerInputType]='accuracy',
    #     iid: Optional[bool]=True,
    #     refit: Optional[RefitType]=True,
    #     cv: Optional[Union[int, Iterable, None]]=None,
    #     verbose: Optional[Union[int, float, bool]]=0,
    #     error_score: Optional[Union[Literal['raise'], int, float]]='raise',
    #     return_train_score: Optional[bool]=False,
    #     scheduler: Optional[Union[SchedulerType, None]]=None,
    #     n_jobs: Optional[Union[int,None]]=None,
    #     cache_cv: Optional[bool]=True
    #     ):


    # fixtures ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

    @staticmethod
    @pytest.fixture(scope='function')
    def _GSTCVDask():
        # 25_04_13 dask_Logistic has a glitch, changed this to sk_Logistic
        return GSTCVDask(
            estimator=sk_LogisticRegression(C=1e-4, tol=1e-6, fit_intercept=True),
            param_grid={},
            thresholds=[0.5],
            cv=2,
            scoring='accuracy',
            refit=False,
            return_train_score=False
        )

    # END fixtures ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

    # estimator ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

    # must be an instance not the class! & be an estimator!

    @pytest.mark.parametrize('not_instantiated',
        (sk_OneHotEncoder, sk_LinearRegression, sk_Ridge, sk_RidgeClassifier,
        sk_LogisticRegression, sk_SGDClassifier, sk_SGDRegressor,
        CalibratedClassifierCV, dask_OneHotEncoder, dask_LinearRegression,
        dask_LogisticRegression)
    )
    def test_rejects_not_instantiated(
        self, _GSTCVDask, not_instantiated, X_da, y_da
    ):

        with pytest.raises(
            TypeError,
            match=f"estimator must be an instance, not the class"
        ):
            _GSTCVDask.set_params(estimator=not_instantiated).fit(X_da, y_da)


    @pytest.mark.parametrize('non_estimator',
        (int, str, list, object, sk_OneHotEncoder, dask_OneHotEncoder,
         sk_CountVectorizer, dask_CountVectorizer)
    )
    def test_rejects_non_estimator(self, _GSTCVDask, non_estimator, X_da, y_da):

        with pytest.raises(AttributeError):
            _GSTCVDask.set_params(estimator=non_estimator()).fit(X_da, y_da)


    @pytest.mark.parametrize('non_dask_classifier',
        (sk_LogisticRegression, )
    )
    def test_warns_on_non_dask_classifiers(
        self, _GSTCVDask, non_dask_classifier, X_da, y_da
    ):

        exp_warn = (f"'{non_dask_classifier().__class__.__name__}' does not "
            f"appear to be a dask classifier.")
        with pytest.warns(match=exp_warn):
            _GSTCVDask.set_params(estimator=non_dask_classifier()).fit(X_da, y_da)


    @pytest.mark.parametrize('non_classifier',
        (sk_LinearRegression, sk_Ridge, sk_SGDRegressor)
    )
    def test_rejects_non_dask_non_classifier(
        self, _GSTCVDask, non_classifier, X_da, y_da
    ):
        with pytest.raises(AttributeError):
            _GSTCVDask.set_params(estimator=non_classifier()).fit(X_da, y_da)


    @pytest.mark.parametrize('dask_non_classifiers', (dask_LinearRegression, ))
    def test_rejects_all_dask_non_classifiers(
        self, _GSTCVDask, dask_non_classifiers, X_da, y_da
    ):

        # must be an instance not the class! & be a classifier!
        with pytest.raises(AttributeError):
            _GSTCVDask.set_params(estimator=dask_non_classifiers()).fit(X_da, y_da)


    def test_accepts_all_dask_classifiers(self, _GSTCVDask, X_da, y_da, _client):

        # must be an instance not the class! & be a classifier!

        # AS OF 24_10_28 THIS IS THE ONLY ONE WORKING ON WINDOWS
        _GSTCVDask.set_params(estimator=sk_LogisticRegression()).fit(X_da, y_da)

    # END estimator ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **


    # iid ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

    # iid: Optional[bool]=True,

    @pytest.mark.parametrize('junk_iid',
        (0, 1, 3.14, None, min, 'junk', [0,1], (0,1), {0,1}, {'a':1}, lambda x: x)
    )
    def test_rejects_all_non_bool(self, _GSTCVDask, junk_iid, X_da, y_da):
        with pytest.raises(TypeError):
            _GSTCVDask.set_params(iid=junk_iid).fit(X_da, y_da)


    @pytest.mark.parametrize('good_iid', (True, False))
    def test_accepts_bool(self, _GSTCVDask, good_iid, X_da, y_da, _client):

        assert isinstance(
            _GSTCVDask.set_params(iid=good_iid).fit(X_da, y_da),
            type(_GSTCVDask)
        )

        assert _GSTCVDask.get_params(deep=True)['iid'] is good_iid


    # END iid ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **


    # cache_cv ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **
    # cache_cv: Optional[bool]=True

    @pytest.mark.parametrize('junk_cachecv',
        (0, 1, 3.14, None, min, 'junk', [0,1], (0,1), {0,1}, {'a':1}, lambda x: x)
    )
    def test_rejects_all_non_bool(self, _GSTCVDask, junk_cachecv, X_da, y_da):
        with pytest.raises(TypeError):
            _GSTCVDask.set_params(cache_cv=junk_cachecv).fit(X_da, y_da)


    @pytest.mark.parametrize('good_cachecv', (True, False))
    def test_accepts_bool(self, _GSTCVDask, good_cachecv, X_da, y_da, _client):

        assert isinstance(
            _GSTCVDask.set_params(cache_cv=good_cachecv).fit(X_da, y_da),
            type(_GSTCVDask)
        )

        assert _GSTCVDask.get_params(deep=True)['cache_cv'] is good_cachecv


    # END cache_cv ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **


    # scheduler ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

    # scheduler: Optional[Union[SchedulerType, None]]=None,
    @staticmethod
    @pytest.fixture
    def marked_client_class():
        class PyBearClient(Client):
            pass

        return PyBearClient


    def test_none_returns_a_scheduler_instance(
        self, _GSTCVDask, X_da, y_da, _client
    ):

        assert isinstance(
            _GSTCVDask.set_params(scheduler=None, n_jobs=1).fit(X_da, y_da),
            type(_GSTCVDask)
        )

        assert _GSTCVDask.get_params(deep=True)['scheduler'] is None


    def test_original_scheduler_is_returned(
        self, _GSTCVDask, marked_client_class, X_da, y_da
    ):

        assert isinstance(
            _GSTCVDask.set_params(
                scheduler=marked_client_class(),
                n_jobs=1
            ).fit(X_da, y_da),
            type(_GSTCVDask)
        )

        assert isinstance(
            _GSTCVDask.get_params(deep=True)['scheduler'],
            marked_client_class
        )

    # END scheduler ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **








