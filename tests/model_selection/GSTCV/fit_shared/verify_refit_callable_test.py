# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

import pytest
from copy import deepcopy

from pybear.model_selection.GSTCV._fit_shared._verify_refit_callable import \
    _verify_refit_callable



class TestVerifyRefitCallable:

    #     def _verify_refit_callable(
    #         refit_callable: RefitCallableType,
    #         DUMMY_CV_RESULTS: CVResultsType
    #         ) -> None

    @pytest.mark.parametrize('junk_callable', (0, True, 'junk', [0,1], (0,1)))
    @pytest.mark.parametrize(
        '_cv_results_template',
        [{
            '_n_splits': 3,
            '_n_rows': 6,
            '_scorer_names': ['accuracy', 'balanced_accuracy'],
            '_grids': [{'param_1':[1,2,3], 'param_2':[True, False]}],
            '_return_train_score': True,
            '_fill_param_columns': True
        }],
        indirect=True
    )
    def test_excepts_on_non_callable(self, _cv_results_template, junk_callable):

        with pytest.raises(ValueError):
            _verify_refit_callable(
                junk_callable,
                deepcopy(_cv_results_template)
            )


    @pytest.mark.parametrize('bad_callable',
        (
            lambda x: -1,
            lambda x: 200_000,
            lambda x: 'trash',
            lambda x: [0,1],
            lambda x: min,
        )
    )
    @pytest.mark.parametrize(
        '_cv_results_template',
        [{
            '_n_splits': 3,
            '_n_rows': 6,
            '_scorer_names': ['accuracy', 'balanced_accuracy'],
            '_grids': [{'param_1':[1,2,3], 'param_2':[True, False]}],
            '_return_train_score': True,
            '_fill_param_columns': True
        }],
        indirect=True
    )
    def test_excepts_on_bad_output(self, _cv_results_template, bad_callable):

        with pytest.raises(ValueError):
            _verify_refit_callable(
                bad_callable,
                deepcopy(_cv_results_template)
            )












































