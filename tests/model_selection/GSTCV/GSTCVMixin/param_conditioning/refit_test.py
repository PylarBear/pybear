# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

from sklearn.metrics import accuracy_score, balanced_accuracy_score

from pybear.model_selection.GSTCV._GSTCVMixin._param_conditioning._refit \
    import _cond_refit



class TestCondRefit:

    # def _cond_refit(
    #     _refit: RefitType,
    #     _scorer: ScorerWIPType
    # ) -> RefitType:


    one_scorer = {'accuracy': accuracy_score}


    two_scorers = {
        'accuracy': accuracy_score,
        'balanced_accuracy': balanced_accuracy_score
    }


    @pytest.mark.parametrize('n_scorers', (one_scorer, two_scorers))
    @pytest.mark.parametrize('junk_refit',
        (0, 1, 3.14, None, [0,1], (0,1), {0,1}, {'a':1})
    )
    def test_reject_junk_refit(self, n_scorers, junk_refit):
        with pytest.raises(Exception):
            _cond_refit(junk_refit, n_scorers)


    @pytest.mark.parametrize('n_scorers', (one_scorer, two_scorers))
    @pytest.mark.parametrize('_callable',
        (lambda X: 0, lambda X: len(X['params'])-1, lambda X: 'trash')
    )
    def test_accepts_callable(self, n_scorers, _callable):
        assert _cond_refit(_callable, n_scorers) == _callable


    @pytest.mark.parametrize('n_scorers', (one_scorer, two_scorers))
    def test_accepts_False(self, n_scorers):

            assert _cond_refit(False, n_scorers) is False


    @pytest.mark.parametrize('n_scorers', (one_scorer,))
    def test_single_accepts_true(self, n_scorers):
        assert _cond_refit(True, n_scorers) == 'score'


    @pytest.mark.parametrize('n_scorers', (two_scorers,))
    def test_accepts_good_strings(self, n_scorers):
        if len(n_scorers) == 1:
            assert _cond_refit('ACCURACY', n_scorers) == 'score'
        if len(n_scorers) == 2:
            assert _cond_refit('BALANCED_ACCURACY', n_scorers) == 'balanced_accuracy'






