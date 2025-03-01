# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.preprocessing._MinCountTransformer._validation._core_bool_val import \
    _core_bool_val

import pytest



class TestCoreBoolVal:


    @pytest.mark.parametrize('junk_param',
        (-1, 0, 1, 2.7, True, False, None, [0,1], (1,), {'a':1}, lambda x: x)
    )
    def test_rejects_junk_param(self, junk_param):

        with pytest.raises(AssertionError):
            _core_bool_val(junk_param, True)



    @pytest.mark.parametrize('good_param', ('eggs', 'bacon', 'waffles', 'coffee'))
    def test_accepts_good_param(self, good_param):

            _core_bool_val(good_param, True)


    @pytest.mark.parametrize('junk_value',
        (-1, 0, 1, 2.7, 'trash', None, [0,1], (1,), {'a':1}, lambda x: x)
    )
    def test_rejects_junk_value(self, junk_value):

        with pytest.raises(TypeError):
            _core_bool_val('some_param', junk_value)


    @pytest.mark.parametrize('good_value', (True, False))
    def test_accepts_good_param(self, good_value):

            _core_bool_val('some_param', good_value)







