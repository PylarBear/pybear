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

# X1_None_p     time = 0.073 +/- 0.005 sec; mem = 0.000 +/- 0.000 MB
# X1_None_t     time = 0.087 +/- 0.002 sec; mem = 0.000 +/- 0.000 MB

# X1_1_p        time = 0.062 +/- 0.003 sec; mem = 0.000 +/- 0.000 MB
# X1_1_t        time = 0.069 +/- 0.003 sec; mem = 0.000 +/- 0.000 MB

# X1_2_p        time = 0.150 +/- 0.009 sec; mem = 0.000 +/- 0.000 MB
# X1_2_t        time = 0.071 +/- 0.003 sec; mem = 0.000 +/- 0.000 MB

# X1_3_p        time = 0.154 +/- 0.005 sec; mem = 0.000 +/- 0.000 MB
# X1_3_t        time = 0.088 +/- 0.008 sec; mem = 0.000 +/- 0.000 MB

# X1_4_p        time = 0.163 +/- 0.017 sec; mem = 0.000 +/- 0.000 MB
# X1_4_t        time = 0.094 +/- 0.006 sec; mem = 0.000 +/- 0.000 MB

# X1_-1_p       time = 0.165 +/- 0.007 sec; mem = 0.000 +/- 0.000 MB
# X1_-1_t       time = 0.083 +/- 0.008 sec; mem = 0.000 +/- 0.000 MB

# X2_None_p     time = 13.041 +/- 1.041 sec; mem = 0.000 +/- 0.000 MB
# X2_None_t     time = 10.333 +/- 0.196 sec; mem = 0.000 +/- 0.000 MB

# X2_1_p        time = 11.076 +/- 0.904 sec; mem = 0.000 +/- 0.000 MB
# X2_1_t        time = 10.399 +/- 0.192 sec; mem = 0.000 +/- 0.000 MB

# X2_2_p        time = 9.879 +/- 1.950 sec; mem = 0.000 +/- 0.000 MB
# X2_2_t        time = 29.518 +/- 0.835 sec; mem = 0.000 +/- 0.000 MB

# X2_3_p        time = 9.865 +/- 1.381 sec; mem = 0.000 +/- 0.000 MB
# X2_3_t        time = 28.273 +/- 0.631 sec; mem = 0.000 +/- 0.000 MB

# X2_4_p        time = 9.328 +/- 0.949 sec; mem = 0.000 +/- 0.000 MB
# X2_4_t        time = 27.683 +/- 0.107 sec; mem = 0.000 +/- 0.000 MB

# X2_-1_p       time = 8.394 +/- 0.635 sec; mem = 0.000 +/- 0.000 MB
# X2_-1_t       time = 29.667 +/- 2.793 sec; mem = 0.000 +/- 0.000 MB








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


















