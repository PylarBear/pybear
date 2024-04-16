# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pytest
import numpy as np
from pybear.sparse_dict._utils import get_shape

NAMES = ['None']
for _ in range(10):
    if _ < 5:
        NAMES.append("NP_ARRAY")
    elif _ >= 5:
        NAMES.append(f"SPARSE_DICT{_ + 1}")


OBJECTS = (
            None,
            [],
            [[], []],
            [2, 3, 4],
            [[2, 3, 4]],
            [[1, 2], [3, 4]],
            {},
            {0: {}, 1: {}},
            {0: 2, 1: 3, 2: 4},
            {0: {0: 2, 1: 3, 2: 4}},
            {0: {0: 1, 1: 2}, 1: {0: 3, 1: 4}}
           )
KEY = (
(), (0,), (2, 0), (3,), (1, 3), (2, 2), (0,), (2, 0), (3,), (1, 3), (2, 2))

for name, _OBJ, _key in zip(NAMES, OBJECTS, KEY):
    _shape = get_shape(name, _OBJ, 'ROW')  # given_orientation == 'ROW'
    if not np.array_equiv(_shape, _key):
        raise Exception(
            f'INCONGRUITY BETWEEN MEASURED SHAPE {_shape} AND ANSWER KEY {_key} FOR {_OBJ}')
    else:
        print(f'\033[92m*** {_OBJ} = {_key}   PASSED! ***\033[0m')

print(f'\n\033[92m*** TESTS COMPLETE.  ALL PASSED. ***\033[0m')


