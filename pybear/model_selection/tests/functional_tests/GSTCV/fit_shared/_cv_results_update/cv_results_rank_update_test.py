# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pytest

from model_selection.GSTCV._master_scorer_dict import master_scorer_dict

from model_selection.GSTCV._fit_shared._cv_results_update._cv_results_rank_update \
    import _cv_results_rank_update




class TestCVResultsRankUpdateTest:


    @staticmethod
    def _scorers():
        return ['accuracy', 'balanced_accuracy'] #, 'recall', 'f1']


    @staticmethod
    @pytest.fixture
    def scorers():
        return ['accuracy', 'balanced_accuracy']# , 'recall', 'f1']


    @pytest.mark.parametrize(
        '_cv_results_template',
        [{
            '_n_splits': 2,
            '_n_rows': 6,
            '_scorer_names': _scorers(),
            '_grids': [{'param_1':[1,2,3], 'param_2':[True, False]}],
            '_return_train_score': True,
            '_fill_param_columns': True
        }],
        indirect=True
    )
    def test_accuracy(self, _cv_results_template, scorers):
        print(
            f"_cv_results_template['mean_fit_time'] = {_cv_results_template['mean_fit_time']}")
        print(
            f"_cv_results_template['std_fit_time'] = {_cv_results_template['std_fit_time']}")
        print(
            f"_cv_results_template['mean_score_time'] = {_cv_results_template['mean_score_time']}")
        print(
            f"_cv_results_template['std_score_time'] = {_cv_results_template['std_score_time']}")
        print(
            f"_cv_results_template['param_param_1'] = {_cv_results_template['param_param_1']}")
        print(
            f"_cv_results_template['param_param_2'] = {_cv_results_template['param_param_2']}")
        print(
            f"_cv_results_template['params'] = {_cv_results_template['params']}")
        print(
            f"_cv_results_template['best_threshold_accuracy'] = {_cv_results_template['best_threshold_accuracy']}")
        print(
            f"_cv_results_template['split0_test_accuracy'] = {_cv_results_template['split0_test_accuracy']}")
        print(
            f"_cv_results_template['split1_test_accuracy'] = {_cv_results_template['split1_test_accuracy']}")
        print(
            f"_cv_results_template['mean_test_accuracy'] = {_cv_results_template['mean_test_accuracy']}")
        print(
            f"_cv_results_template['std_test_accuracy'] = {_cv_results_template['std_test_accuracy']}")
        print(
            f"_cv_results_template['rank_test_accuracy'] = {_cv_results_template['rank_test_accuracy']}")
        print(
            f"_cv_results_template['split0_train_accuracy'] = {_cv_results_template['split0_train_accuracy']}")
        print(
            f"_cv_results_template['split1_train_accuracy'] = {_cv_results_template['split1_train_accuracy']}")
        print(
            f"_cv_results_template['mean_train_accuracy'] = {_cv_results_template['mean_train_accuracy']}")
        print(
            f"_cv_results_template['std_train_accuracy'] = {_cv_results_template['std_train_accuracy']}")
        print(
            f"_cv_results_template['best_threshold_balanced_accuracy'] = {_cv_results_template['best_threshold_balanced_accuracy']}")
        print(
            f"_cv_results_template['split0_test_balanced_accuracy'] = {_cv_results_template['split0_test_balanced_accuracy']}")
        print(
            f"_cv_results_template['split1_test_balanced_accuracy'] = {_cv_results_template['split1_test_balanced_accuracy']}")
        print(
            f"_cv_results_template['mean_test_balanced_accuracy'] = {_cv_results_template['mean_test_balanced_accuracy']}")
        print(
            f"_cv_results_template['std_test_balanced_accuracy'] = {_cv_results_template['std_test_balanced_accuracy']}")
        print(
            f"_cv_results_template['rank_test_balanced_accuracy'] = {_cv_results_template['rank_test_balanced_accuracy']}")
        print(
            f"_cv_results_template['split0_train_balanced_accuracy'] = {_cv_results_template['split0_train_balanced_accuracy']}")
        print(
            f"_cv_results_template['split1_train_balanced_accuracy'] = {_cv_results_template['split1_train_balanced_accuracy']}")
        print(
            f"_cv_results_template['mean_train_balanced_accuracy'] = {_cv_results_template['mean_train_balanced_accuracy']}")
        print(f"_cv_results_template['std_train_balanced_accuracy'] = {_cv_results_template['std_train_balanced_accuracy']}")





        out = _cv_results_rank_update(
            _scorer={k:v for k,v in master_scorer_dict.items() if k in scorers},
            _cv_results=_cv_results_template
        )





    # for scorer_suffix in _scorer:
    #
    #     if f'rank_test_{scorer_suffix}' not in _cv_results:
    #         raise ValueError(f"appending tests scores to a column in cv_results_ "
    #             f"that doesnt exist but should (rank_test_{scorer_suffix})")
    #
    #     _ = _cv_results[f'mean_test_{scorer_suffix}']
    #     _cv_results[f'rank_test_{scorer_suffix}'] = \
    #         len(_) - ss.rankdata(_, method='max') + 1
    #     del _
    #
    # return _cv_results






























