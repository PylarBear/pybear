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


@pytest.fixture(scope='module')
def _dupls(_shape):
    # _dupl must be intermingled like [[0,8],[1,9]], not [[0,1],[8,9]]
    # for TestManyPartialFitsEqualOneBigFit to catch 'random' putting
    # out different columns over a sequence of transforms
    return [[0,_shape[1]-2], [1, _shape[1]-1]]


@pytest.fixture(scope='module')
def X_np(_X_factory, _dupls, _shape):
    return _X_factory(
        _dupl=_dupls,
        _has_nan=False,
        _dtype='flt',
        _shape=_shape
    )


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
        'n_jobs': 1,     # leave set at 1 because of confliction
        'job_size': 20
    }




