# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.preprocessing.InterceptManager._partial_fit._set_attributes import \
    _set_attributes

import numpy as np

import pytest





class TestSetAttributes:


    # def _set_attributes(
    #     constant_columns_: dict[int, any],
    #     _instructions: dict[str, Union[None, Iterable[int]]],
    #     _n_features: int
    # ) -> tuple[dict[int, any], dict[int, any], npt.NDArray[bool]]:


    @staticmethod
    @pytest.fixture(scope='module')
    def _instructions():
        return {
            'keep': None,
            'delete': [1, 3, 5],
            'add': {'Intercept', 1}
        }


    @staticmethod
    @pytest.fixture(scope='module')
    def _constant_columns():
        return {1: 0, 3: 1, 5:np.nan}


    def test_accuracy(self, _instructions, _constant_columns):

        out_kept_columns, out_removed_columns, out_column_mask = \
            _set_attributes(
                _constant_columns,
                _instructions,
                _n_features=6
            )

        assert out_kept_columns == {}
        assert out_removed_columns == {1: 0, 3: 1, 5:np.nan}
        assert np.array_equal(out_column_mask, [1,0,1,0,1,0])















