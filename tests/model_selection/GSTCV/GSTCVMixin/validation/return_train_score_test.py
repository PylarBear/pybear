# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pytest

from pybear.model_selection.GSTCV._GSTCVMixin._validation._return_train_score \
    import _validate_return_train_score



class TestValidateReturnTrainScore:

    @pytest.mark.parametrize('non_bool',
        (-1, 0, 3.14, None, 'junk', min, [0,1], (0,1), {0,1}, {'a':1}, lambda x: x)
    )
    def test_rejects_non_bool(self, non_bool):
        with pytest.raises(TypeError):
            _validate_return_train_score(non_bool)


    @pytest.mark.parametrize('good_bool', (True, False))
    def test_accepts_bool(self, good_bool):
        assert _validate_return_train_score(good_bool) is good_bool




