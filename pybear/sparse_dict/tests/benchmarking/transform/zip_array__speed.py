# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

from pybear.utils import time_memory_benchmark as tmb
from pybear.sparse_dict._transform import zip_array, zip_array2
import numpy as np


# LINUX TIME TRIALS 24_04_26_13_00_00

# NO joblib IS THE CLEAR WINNER, SURPRISINGLY

tmb(
    # NO PARALLELISM
    ('zip_array', zip_array, [np.random.randint(0,2,(5_000,5_000))], {'dtype':np.uint8}),
    # zip_array      time = 4.920 +/- 0.136 sec; mem = 987.667 +/- 9.741 MB
    # zip_array      time = 5.376 +/- 0.148 sec; mem = 1,079.667 +/- 1.247 MB
    # zip_array      time = 4.934 +/- 0.029 sec; mem = 1,111.667 +/- 0.471 MB

    # 'zip_array2' was a for loop around joblib.processes to use parallel on batches of inner dicts
    # ('zip_array2', zip_array2, [np.random.randint(0,2,(5_000,5_000))], {'dtype':np.uint8}),
    #  THIS REALLY IS TRUE & REPEATABLE, NO IDEA WHY
    #  zip_array2   time = 112.766 +/- 2.190 sec; mem = 0.000 +/- 0.000 MB

    # 'zip_array3' was joblib.threads in one shot to make inner dicts
    #  ('zip_array3', zip_array3, [np.random.randint(0,2,(5_000,5_000))], {'dtype':np.uint8}),
    #  zip_array3   time = 6.850 +/- 0.089 sec; mem = 1,089.333 +/- 3.300 MB

    # 'zip_array4' was joblib.processes in one shot to make inner dicts
    # ('zip_array4', zip_array4, [np.random.randint(0,2,(5_000,5_000))], {'dtype':np.uint8}),
    #  THIS REALLY IS TRUE & REPEATABLE, NO IDEA WHY
    #   zip_array4   time = 98.077 +/- 3.541 sec; mem = 741.667 +/- 8.340 MB
    rest_time=1,
    number_of_trials=5
)














