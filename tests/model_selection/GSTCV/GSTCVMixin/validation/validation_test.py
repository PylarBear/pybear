# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.model_selection.GSTCV._GSTCVMixin._validation._validation \
    import _validation

import numpy as np

import pytest



class TestValidation:


    @pytest.mark.parametrize('_return_train_score', (True, False))
    @pytest.mark.parametrize('_error_score', (np.e, np.nan, 'raise'))
    @pytest.mark.parametrize('_verbose', (0, False, True, 1, 1_000))
    @pytest.mark.parametrize('_cv',
        (None, 2, ((range(3), range(1,4)), (range(2,5), range(3,6))))
    )
    @pytest.mark.parametrize('_n_jobs', (None, -1, 1, 2))
    @pytest.mark.parametrize('_scoring',
        ('accuracy', ['precision', 'recall'], lambda x,y: 0.2835,
         {'user1': lambda x,y: 0, 'user2': lambda x,y: 1})
    )
    def test_accuracy(
        self, _scoring, _n_jobs, _cv, _verbose, _error_score,
        _return_train_score
    ):

        assert _validation(
            _scoring,
            _n_jobs,
            _cv,
            _verbose,
            _error_score,
            _return_train_score
        ) is None





