# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

import numpy as np



@pytest.fixture(scope='session')
def _shape():
    return (20, 10)


@pytest.fixture(scope='session')
def y_np(_shape):
    return np.random.randint(0, 2, _shape[0])


@pytest.fixture(scope='function')
def _kwargs():
    return {
        'keep': 'first',
        'equal_nan': False,   # must be False
        'rtol': 1e-5,
        'atol': 1e-8
    }



