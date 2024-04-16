import numpy as np
from pybear.new_numpy._random import choice
from pybear.utils import time_memory_benchmark as tmb



# PRE-JOBLIB (24_04_11_15_30_00)
# WINDOWS, POOL = 1e8, SELECTION = 10000, replace=False, AVERAGE OF 10 TRIALS.
# 22_11_24 np.random.choice             time = 8.825 +/- 0.041 sec; mem = 0.000 +/- 0.000 MB
# 22_11_24 new_numpy.random.choice      time = 3.774 +/- 0.017 sec; mem = 0.000 +/- 0.000 MB
# 24_04_10 np.random.choice             time = 10.532 +/- 0.672 sec; mem = 0.000 +/- 0.000 MB
# 24_04_10 new_numpy.random.choice      time = 4.599 +/- 0.052 sec; mem = 0.000 +/- 0.000 MB

# POST-JOBLIB - PROCESSES
# WINDOWS, POOL = 1e8, SELECTION = 10000, replace=False, AVERAGE OF 10 TRIALS.
# 24_04_11 np.random.choice            time = 10.178 +/- 0.629 sec; mem = 0.000 +/- 0.000 MB
# 24_04_11 new_numpy.random.choice     time = 3.743 +/- 0.491 sec; mem = 0.625 +/- 1.654 MB

# POST-JOBLIB - THREADS
# WINDOWS, POOL = 1e8, SELECTION = 10000, replace=False, AVERAGE OF 10 TRIALS.
# 24_04_11 np.random.choice            time = 10.087 +/- 0.845 sec; mem = 0.000 +/- 0.000 MB
# 24_04_11 new_numpy.random.choice     time = 4.838 +/- 0.080 sec; mem = 0.125 +/- 0.331 MB



# PRE-JOBLIB (24_04_11_15_30_00)
# LINUX, POOL = 1e8, SELECTION = 10000, replace=False, AVERAGE OF 10 TRIALS.
# 24_03_20 np.random.choice             time = 6.508 +/- 0.564 sec; mem = 0.000 +/- 0.000 MB
# 24_03_20 new_numpy.random.choice      time = 2.969 +/- 0.195 sec; mem = 0.000 +/- 0.000 MB
# 24_03_20 np.random.choice             time = 6.163 +/- 0.028 sec; mem = 0.000 +/- 0.000 MB
# 24_03_20 new_numpy.random.choice      time = 2.866 +/- 0.011 sec; mem = 0.000 +/- 0.000 MB






# TIME MEMORY TESTS ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
print(f'\nRunning time & memory tests...')
a = np.arange(int(1e8), dtype=np.int32)
shape = (10000,)

TIME_MEM = tmb(
                ('np.random.choice', np.random.choice, [a, shape], {'replace' :False}),
                ('new_numpy.random.choice', choice, [a, shape], {'replace' :False, 'n_jobs':-1}),
                number_of_trials=10,
                rest_time=2
)












