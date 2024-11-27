# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.preprocessing.InterceptManager.InterceptManager import \
    InterceptManager as IM

from pybear.utilities._benchmarking import time_memory_benchmark as tmb

import numpy as np
import joblib




# this benchmarks fit times for fit with different X shapes and n_jobs



# ubuntu 24.04

# dataset_n_jobs_process/threads

# X1_None_p     time = 0.072 +/- 0.010 sec; mem = 0.000 +/- 0.000 MB
# X1_1_p        time = 0.060 +/- 0.001 sec; mem = 0.000 +/- 0.000 MB
# X1_2_p        time = 0.143 +/- 0.014 sec; mem = 0.000 +/- 0.000 MB
# X1_3_p        time = 0.169 +/- 0.013 sec; mem = 0.000 +/- 0.000 MB
# X1_4_p        time = 0.172 +/- 0.007 sec; mem = 0.000 +/- 0.000 MB
# X1_-1_p       time = 0.170 +/- 0.006 sec; mem = 0.000 +/- 0.000 MB
# X2_None_p     time = 11.683 +/- 1.035 sec; mem = 0.000 +/- 0.000 MB
# X2_1_p        time = 11.621 +/- 1.058 sec; mem = 0.000 +/- 0.000 MB
# X2_2_p        time = 8.887 +/- 0.861 sec; mem = 0.333 +/- 0.471 MB
# X2_3_p        time = 8.865 +/- 0.985 sec; mem = 0.000 +/- 0.000 MB
# X2_4_p        time = 8.942 +/- 1.103 sec; mem = 0.000 +/- 0.000 MB
# X2_-1_p       time = 8.857 +/- 1.045 sec; mem = 0.000 +/- 0.000 MB
# X1_None_t     time = 0.070 +/- 0.011 sec; mem = 0.000 +/- 0.000 MB
# X1_1_t        time = 0.053 +/- 0.004 sec; mem = 0.000 +/- 0.000 MB
# X1_2_t        time = 0.069 +/- 0.001 sec; mem = 0.000 +/- 0.000 MB
# X1_3_t        time = 0.076 +/- 0.001 sec; mem = 0.000 +/- 0.000 MB
# X1_4_t        time = 0.073 +/- 0.003 sec; mem = 0.000 +/- 0.000 MB
# X1_-1_t       time = 0.079 +/- 0.005 sec; mem = 0.000 +/- 0.000 MB
# X2_None_t     time = 11.610 +/- 1.049 sec; mem = 0.000 +/- 0.000 MB
# X2_1_t        time = 11.563 +/- 1.047 sec; mem = 0.000 +/- 0.000 MB
# X2_2_t        time = 29.145 +/- 5.898 sec; mem = 0.000 +/- 0.000 MB
# X2_3_t        time = 31.954 +/- 6.462 sec; mem = 0.000 +/- 0.000 MB
# X2_4_t        time = 30.918 +/- 5.360 sec; mem = 0.000 +/- 0.000 MB
# X2_-1_t       time = 31.056 +/- 5.568 sec; mem = 0.000 +/- 0.000 MB








shape1 = (50_000, 10)
shape2 = (10, 50_000)


X1 = np.random.randint(0, 10, shape1).astype(np.uint8)
X1[:, np.random.choice(range(shape1[1]), shape1[1]//5, replace=False)] = 1


X2 = np.random.randint(0, 10, shape2).astype(np.uint8)
X2[:, np.random.choice(range(shape2[1]), shape2[1]//100, replace=False)] = 1




def fit_benchmark_fxn(
    X: tuple[int, int] = None,
    backend: str = 'm',
    _n_jobs: int = None
):

    _IM = IM(
        keep='first',
        equal_nan=True,
        rtol=1e-5,
        atol=1e-8,
        n_jobs=None
    )

    backend = 'multiprocessing' if backend == 'm' else 'threading'

    with joblib.parallel_config(backend = backend, n_jobs=_n_jobs):
        _IM.fit(X)





results = tmb(
    ('X1_None_p', fit_benchmark_fxn, [], {'X':X1, 'backend': 'm', '_n_jobs':None}),
    ('X1_1_p', fit_benchmark_fxn, [], {'X':X1, 'backend': 'm', '_n_jobs':1}),
    ('X1_2_p', fit_benchmark_fxn, [], {'X':X1, 'backend': 'm', '_n_jobs':2}),
    ('X1_3_p', fit_benchmark_fxn, [], {'X':X1, 'backend': 'm', '_n_jobs':3}),
    ('X1_4_p', fit_benchmark_fxn, [], {'X':X1, 'backend': 'm', '_n_jobs':4}),
    ('X1_-1_p', fit_benchmark_fxn, [], {'X':X1, 'backend': 'm', '_n_jobs':-1}),
    ('X2_None_p', fit_benchmark_fxn, [], {'X':X2, 'backend': 'm', '_n_jobs':None}),
    ('X2_1_p', fit_benchmark_fxn, [], {'X':X2, 'backend': 'm', '_n_jobs':1}),
    ('X2_2_p', fit_benchmark_fxn, [], {'X':X2, 'backend': 'm', '_n_jobs':2}),
    ('X2_3_p', fit_benchmark_fxn, [], {'X':X2, 'backend': 'm', '_n_jobs':3}),
    ('X2_4_p', fit_benchmark_fxn, [], {'X':X2, 'backend': 'm', '_n_jobs':4}),
    ('X2_-1_p', fit_benchmark_fxn, [], {'X':X2, 'backend': 'm', '_n_jobs':-1}),
    ('X1_None_t', fit_benchmark_fxn, [], {'X':X1, 'backend': 't', '_n_jobs':None}),
    ('X1_1_t', fit_benchmark_fxn, [], {'X':X1, 'backend': 't', '_n_jobs':1}),
    ('X1_2_t', fit_benchmark_fxn, [], {'X':X1, 'backend': 't', '_n_jobs':2}),
    ('X1_3_t', fit_benchmark_fxn, [], {'X':X1, 'backend': 't', '_n_jobs':3}),
    ('X1_4_t', fit_benchmark_fxn, [], {'X':X1, 'backend': 't', '_n_jobs':4}),
    ('X1_-1_t', fit_benchmark_fxn, [], {'X':X1, 'backend': 't', '_n_jobs':-1}),
    ('X2_None_t', fit_benchmark_fxn, [], {'X': X2, 'backend': 't', '_n_jobs': None}),
    ('X2_1_t', fit_benchmark_fxn, [], {'X':X2, 'backend': 't', '_n_jobs':1}),
    ('X2_2_t', fit_benchmark_fxn, [], {'X':X2, 'backend': 't', '_n_jobs':2}),
    ('X2_3_t', fit_benchmark_fxn, [], {'X':X2, 'backend': 't', '_n_jobs':3}),
    ('X2_4_t', fit_benchmark_fxn, [], {'X':X2, 'backend': 't', '_n_jobs':4}),
    ('X2_-1_t', fit_benchmark_fxn, [], {'X':X2, 'backend': 't', '_n_jobs':-1}),
    number_of_trials=5,
    rest_time=1,
    verbose=1
)


















