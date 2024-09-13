# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

from pybear.model_selection.autogridsearch._autogridsearch_wrapper. \
    _build_first_grid_from_params import _build




class TestBuild:

    # There is no _validation on the _build module

    def test_accuracy(self):

        _params = {
            'string' : [['a','b','c'], 3, 'string'],
            'num': [[1,2,3,4], [4,4,4,4,4], 'fixed_integer']
        }

        assert _build(_params) == {0: {'string':['a','b','c'], 'num':[1,2,3,4]}}







