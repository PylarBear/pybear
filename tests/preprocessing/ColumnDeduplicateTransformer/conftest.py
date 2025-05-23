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
        'do_not_drop': None,
        'conflict': 'raise',
        'rtol': 1e-5,
        'atol': 1e-8,
        'equal_nan': False,
        'n_jobs': 1     # leave set at 1 because of confliction
    }


@pytest.fixture(scope='module')
def _dupl1():
    return [
        [0, 7],
        [1, 5, 8]
    ]


@pytest.fixture(scope='module')
def _dupl2():
    return []


@pytest.fixture(scope='module')
def _dupl3():
    return [
        [0, 7, 9],
        [1, 5, 6, 8]
    ]


@pytest.fixture(scope='module')
def _dupl4():
    return [[0, 4, 7]]





