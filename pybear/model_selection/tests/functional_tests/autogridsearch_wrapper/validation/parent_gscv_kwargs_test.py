# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pytest

from model_selection.autogridsearch._autogridsearch_wrapper._validation. \
    _parent_gscv_kwargs import _val_parent_gscv_kwargs

from sklearn.linear_model import LogisticRegression as skl_logistic
from sklearn.model_selection import GridSearchCV as skl_GridSearchCV

from dask_ml.model_selection import GridSearchCV as dask_GridSearchCV
from dask_ml.linear_model import LogisticRegression as dask_logistic




class TestValParentGSCVKwargs:

    # sklearn ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
    def test_rejects_bad_sklearn_GSCV_kwargs(self):

        _estimator = skl_logistic()

        _GSCV_parent = skl_GridSearchCV

        with pytest.raises(ValueError):
            _val_parent_gscv_kwargs(
                _estimator,
                _GSCV_parent,
                {'aaa': True, 'bbb': 1.5}
            )


    def test_accepts_good_sklearn_GSCV_kwargs(self):

        _estimator = skl_logistic()

        _GSCV_parent = skl_GridSearchCV

        _val_parent_gscv_kwargs(
            _estimator,
            _GSCV_parent,
            {'scoring':'accuracy', 'n_jobs':-1, 'cv':5}
        )


    @pytest.mark.parametrize('_refit', (True, False))
    def test_sklearn_GSCV_indifferent_to_refit(self, _refit):

        _estimator = skl_logistic()

        _GSCV_parent = skl_GridSearchCV

        _val_parent_gscv_kwargs(
            _estimator,
            _GSCV_parent,
            {'refit': _refit}
        )

    # END sklearn ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *



    # dask ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **
    def test_rejects_bad_dask_GSCV_kwargs(self):

        _estimator = dask_logistic()

        _GSCV_parent = dask_GridSearchCV

        with pytest.raises(ValueError):
            _val_parent_gscv_kwargs(
                _estimator,
                _GSCV_parent,
                {'junk': True, 'trash': 'balanced_accuracy'}
            )


    def test_accepts_good_dask_GSCV_kwargs(self):

        _estimator = dask_logistic()

        _GSCV_parent = dask_GridSearchCV

        _val_parent_gscv_kwargs(
            _estimator,
            _GSCV_parent,
            {'scoring':'accuracy', 'n_jobs':-1, 'cv':5}
        )


    def test_dask_GSCV_refit_accept_true_reject_false(self):

        _estimator = dask_logistic()

        _GSCV_parent = dask_GridSearchCV

        _val_parent_gscv_kwargs(
            _estimator,
            _GSCV_parent,
            {'refit': True}
        )

        with pytest.raises(AttributeError):
            _val_parent_gscv_kwargs(
                _estimator,
                _GSCV_parent,
                {'refit': False}
            )

    # END sklearn ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *























