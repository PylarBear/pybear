# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.preprocessing.MinCountTransformer._validation.\
    _val_ign_cols_hab_callable import _val_ign_cols_hab_callable

import pytest



class TestValIgnColsHabCallable:

    @pytest.mark.parametrize('bad_output',
        (0, min, 'trash', {'a': 1}, None, True, False, lambda x: x)
    )
    def test_rejects_bad_output(self, bad_output):

        with pytest.raises(TypeError):
            _val_ign_cols_hab_callable(
                bad_output,
                'ignore_columns'
            )


    def test_accepts_list_of_ints(self):
        _val_ign_cols_hab_callable(
            [0, 2, 4, 6],
            'handle_as_bool'
        )



    def test_accepts_list_of_strs(self):
        _val_ign_cols_hab_callable(
            ['a', 'b', 'c', 'd'],
            'ignore_columns'
        )














