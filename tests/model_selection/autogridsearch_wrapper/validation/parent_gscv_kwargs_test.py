# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

import numpy as np

from pybear.model_selection.autogridsearch._autogridsearch_wrapper. \
    _validation._parent_gscv_kwargs import _val_parent_gscv_kwargs

from pybear.model_selection import GSTCV, GSTCVDask

from sklearn.model_selection import GridSearchCV as sk_GridSearchCV

from dask_ml.model_selection import GridSearchCV as dask_GridSearchCV



class TestValParentGSCVKwargs:


    # common to sklearn & dask ** * ** * ** * ** * ** * ** * ** * ** * **

    def test_blocks_refit_False_and_multiple_scorers(self):

        with pytest.raises(AttributeError):
            _val_parent_gscv_kwargs(
                dask_GridSearchCV,
                {'refit': False, 'scoring': ['accuracy', 'balanced_accuracy']}
            )


    def test_converts_scorer_list_of_one_to_str(self):

        out = _val_parent_gscv_kwargs(
            sk_GridSearchCV,
            {'refit': False, 'scoring': ['accuracy']}
        )

        assert out['scoring'] == 'accuracy'


    # END common to sklearn & dask ** * ** * ** * ** * ** * ** * ** * **


    # sklearn ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
    def test_rejects_bad_sklearn_GSCV_kwargs(self):

        _GSCV_parent = sk_GridSearchCV

        with pytest.raises(ValueError):
            _val_parent_gscv_kwargs(
                _GSCV_parent,
                {'aaa': True, 'bbb': 1.5}
            )


    def test_accepts_good_sklearn_GSCV_kwargs(self):

        _GSCV_parent = sk_GridSearchCV

        _val_parent_gscv_kwargs(
            _GSCV_parent,
            {'scoring':'accuracy', 'n_jobs':-1, 'cv':5}
        )


    @pytest.mark.parametrize('_refit', (True, False, ['accuracy'], lambda x: x))
    def test_sklearn_GSCV_indifferent_to_refit(self, _refit):

        _GSCV_parent = sk_GridSearchCV

        _val_parent_gscv_kwargs(
            _GSCV_parent,
            {'refit': _refit}
        )

    # END sklearn ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *



    # dask ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **
    def test_rejects_bad_dask_GSCV_kwargs(self):

        _GSCV_parent = dask_GridSearchCV

        with pytest.raises(ValueError):
            _val_parent_gscv_kwargs(
                _GSCV_parent,
                {'junk': True, 'trash': 'balanced_accuracy'}
            )


    def test_accepts_good_dask_GSCV_kwargs(self):

        _GSCV_parent = dask_GridSearchCV

        _val_parent_gscv_kwargs(
            _GSCV_parent,
            {'scoring':'accuracy', 'n_jobs':-1, 'cv':5}
        )


    @pytest.mark.parametrize('_refit',
        (True, 'accuracy', lambda x: x, False)
    )
    def test_dask_GSCV_refit_accepts_true_str_fxn_rejects_false(self, _refit):

        if _refit is False:
            with pytest.raises(AttributeError):
                _val_parent_gscv_kwargs(
                    dask_GridSearchCV,
                    {'refit': False}
                )

        else:
            assert _val_parent_gscv_kwargs(
                dask_GridSearchCV,
                {'refit': _refit}
            ) == {'refit': _refit}

    # END dask ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *



    # GSTCV(Dask) ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

    @pytest.mark.parametrize('gscv_parent', (GSTCV, GSTCVDask))
    def test_accepts_threshold_kwarg(self, gscv_parent):

        kwargs = {'thresholds': np.linspace(0, 1, 3)}

        out = _val_parent_gscv_kwargs(
            gscv_parent,
            kwargs
        )

        assert np.array_equiv(out['thresholds'], kwargs['thresholds'])


    # END GSTCV(Dask) ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **












