# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#





from pybear.preprocessing.InterceptManager._partial_fit._merge_constants import (
    _merge_constants
)

import numpy as np

import pytest




class TestMergeConstants:


    def test_accuracy(self):

        _old_constants = {0:1, 4: 0, 5: 0}

        # one column dropped out
        _new_constants = {0:1, 5: 0}
        assert _merge_constants(_old_constants, _new_constants) == \
               {0: 1, 5: 0}

        # new columns of constants, with overlap
        _new_constants = {0: 1, 2: np.nan, 6: 0}
        assert _merge_constants(_old_constants, _new_constants) == \
               {0: 1}

        # new columns of constants, no overlap
        _new_constants = {11: 1, 12: 4, 13: 0}
        assert _merge_constants(_old_constants, _new_constants) == \
               {}

        # same columns, value changed
        _new_constants = {0:1, 4: 1, 5:0}
        assert _merge_constants(_old_constants, _new_constants) == \
               {0:1, 5:0}

        # empty new constants
        _new_constants = {}
        assert _merge_constants(_old_constants, _new_constants) == {}

        # v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^

        # empty old constants --- like on first fit
        _old_constants = {}

        NEW_CONSTANTS = (
            {0: 1, 5: 0},
            {0: 1, 2: np.nan, 6: 0},
            {11: 1, 12: 4, 13: 0},
            {0: 1, 4: 1, 5: 0},
            {}
        )

        for _new_constants in NEW_CONSTANTS:

            out = _merge_constants(_old_constants, _new_constants)

            # need to do this the hard way because of np.nan

            str_out_keys = list(map(str, out.keys()))
            str_out_values = list(map(str, out.values()))
            str_new_keys = list(map(str, _new_constants.keys()))
            str_new_values = list(map(str, _new_constants.values()))


            assert np.array_equal(str_out_keys, str_new_keys)
            assert np.array_equal(str_out_values, str_new_values)












