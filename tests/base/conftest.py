# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#




import uuid
import numpy as np

import pytest



@pytest.fixture(scope='session')
def _master_columns():
    _cols = 200   # do not change this, this gives surplus over columns in _shape
    while True:
        _ = [str(uuid.uuid4())[:4] for _ in range(_cols)]
        if len(np.unique(_)) == len(_):
            return np.array(_, dtype='<U4')






