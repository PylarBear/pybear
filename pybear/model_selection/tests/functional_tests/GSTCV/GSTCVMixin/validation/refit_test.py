# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pytest

from sklearn.metrics import accuracy_score, balanced_accuracy_score

from pybear.model_selection.GSTCV._GSTCVMixin._validation._refit import \
    _validate_refit



class TestValidateRefit:

    # def _validate_refit(
    #     _refit: RefitType,
    #     _scorer: ScorerWIPType
    #     ) -> RefitType:

    @staticmethod
    def one_scorer():
        return {'accuracy': accuracy_score}


    @staticmethod
    def two_scorers():
        return {
                'accuracy': accuracy_score,
                'balanced_accuracy': balanced_accuracy_score
        }


    @pytest.mark.parametrize('n_scorers', (one_scorer(), two_scorers()))
    @pytest.mark.parametrize('junk_refit',
        (0, 1, 3.14, [0,1], (0,1), {0,1}, {'a':1})
    )
    def test_reject_junk_refit(self, n_scorers, junk_refit):
        with pytest.raises(TypeError):
            _validate_refit(junk_refit, n_scorers)


    @pytest.mark.parametrize('n_scorers', (one_scorer(), two_scorers()))
    @pytest.mark.parametrize('_callable',
        (lambda X: 0, lambda X: len(X['params'])-1, lambda X: 'trash')
    )
    def test_accepts_callable(self, n_scorers, _callable):
        assert _validate_refit(_callable, n_scorers) == _callable


    @pytest.mark.parametrize('n_scorers', (one_scorer(), two_scorers()))
    @pytest.mark.parametrize('_refit', (None, False))
    def test_accepts_None_and_False(self, n_scorers, _refit):

        if len(n_scorers) == 1:
            assert _validate_refit(_refit, n_scorers) is False

        elif len(n_scorers) == 2:
            exp_warn = (
                f"WHEN MULTIPLE SCORERS ARE USED:\n"
                f"Cannot return a best threshold if refit is False or callable.\n"
                f"If refit is False: best_index_, best_estimator_, best_score_, "
                f"and best_threshold_ are not available.\n"
                f"if refit is callable: best_score_ and best_threshold_ "
                f"are not available.\n"
                f"In either case, access score and threshold info via the "
                f"cv_results_ attribute."
            )

            with pytest.warns(match=exp_warn):
                assert _validate_refit(_refit, n_scorers) is False


    @pytest.mark.parametrize('n_scorers', (one_scorer(),))
    def test_single_accepts_true(self, n_scorers):
        assert _validate_refit(True, n_scorers) == 'score'


    @pytest.mark.parametrize('n_scorers', (two_scorers(),))
    def test_multi_rejects_true(self, n_scorers):
        with pytest.raises(ValueError):
            _validate_refit(True, n_scorers)


    @pytest.mark.parametrize('n_scorers', (one_scorer(), two_scorers()))
    @pytest.mark.parametrize('junk_string', ('trash', 'garbage', 'junk'))
    def test_rejects_junk_strings(self, n_scorers, junk_string):
        with pytest.raises(ValueError):
            _validate_refit(junk_string, n_scorers)


    @pytest.mark.parametrize('n_scorers', (two_scorers(),))
    def test_accepts_good_strings(self, n_scorers):
        if len(n_scorers) == 1:
            assert _validate_refit('ACCURACY', n_scorers) == 'score'
        if len(n_scorers) == 2:
            assert _validate_refit('BALANCED_ACCURACY', n_scorers) == \
                'balanced_accuracy'





































































