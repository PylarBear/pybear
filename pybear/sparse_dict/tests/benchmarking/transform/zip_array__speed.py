# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

from pybear.utils import time_memory_benchmark as tmb
from pybear.sparse_dict._transform import zip_array
import numpy as np


# LINUX TIME TRIALS 24_04_26_13_00_00 WITH NUMPY ARRAYS ONLY

# NO joblib IS THE CLEAR WINNER, SURPRISINGLY

tmb(
    # NO PARALLELISM
    # ('zip_array', zip_array, [np._random_.randint(0,2,(5_000,5_000))], {'dtype':np.uint8}),
    # BEFORE CHANGE FOR py DTYPES ** * ** * ** * ** * ** * ** * ** * ** * ** *
    # zip_array      time = 4.920 +/- 0.136 sec; mem = 987.667 +/- 9.741 MB
    # zip_array      time = 5.376 +/- 0.148 sec; mem = 1,079.667 +/- 1.247 MB
    # zip_array      time = 4.934 +/- 0.029 sec; mem = 1,111.667 +/- 0.471 MB
    # zip_array      time = 5.408 +/- 0.239 sec; mem = 1,115.333 +/- 0.471 MB

    # 'zip_array2' was a for loop around joblib.processes to use parallel on batches of inner dicts
    # ('zip_array2', zip_array2, [np._random_.randint(0,2,(5_000,5_000))], {'dtype':np.uint8}),
    #  THIS REALLY IS TRUE & REPEATABLE, NO IDEA WHY
    #  zip_array2   time = 112.766 +/- 2.190 sec; mem = 0.000 +/- 0.000 MB

    # 'zip_array3' was joblib.threads in one shot to make inner dicts
    #  ('zip_array3', zip_array3, [np._random_.randint(0,2,(5_000,5_000))], {'dtype':np.uint8}),
    #  zip_array3   time = 6.850 +/- 0.089 sec; mem = 1,089.333 +/- 3.300 MB

    # 'zip_array4' was joblib.processes in one shot to make inner dicts
    # ('zip_array4', zip_array4, [np._random_.randint(0,2,(5_000,5_000))], {'dtype':np.uint8}),
    #  THIS REALLY IS TRUE & REPEATABLE, NO IDEA WHY
    #   zip_array4   time = 98.077 +/- 3.541 sec; mem = 741.667 +/- 8.340 MB
    # END BEFORE CHANGE FOR py DTYPES ** * ** * ** * ** * ** * ** * ** * ** * **


    # AFTER CHANGE FOR py DTYPES
    ('zip_array', zip_array, [np.random.randint(0, 2, (5_000, 5_000))], {'dtype': int}),
    # zip_array      time = 5.525 +/- 0.022 sec; mem = 681.667 +/- 1.247 MB
    # zip_array      time = 5.195 +/- 0.015 sec; mem = 713.667 +/- 0.943 MB

    ('zip_array2', zip_array, [np.random.randint(0, 2, (5_000, 5_000))], {'dtype': np.uint8}),
    # zip_array2     time = 5.150 +/- 0.028 sec; mem = 1,063.333 +/- 1.247 MB
    # zip_array2     time = 4.875 +/- 0.023 sec; mem = 1,096.333 +/- 0.471 MB
    rest_time=1,
    number_of_trials=5
)














