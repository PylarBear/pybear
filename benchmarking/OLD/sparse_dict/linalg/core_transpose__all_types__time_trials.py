# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

import numpy as np
import pandas as pd
from utilities._benchmarking import time_memory_benchmark as tmb
from sparse_dict._random_ import _create_random_sparse_dict
from sparse_dict._linalg import (
                                        core_sparse_transpose_brute_force,
                                        core_sparse_transpose_map
)



# LINUX TRIALS 24_05_06



_rows = 5_000
_columns = 5_000
_sparsity = 50
_rest_time = 2
_num_trials = 5

def build_df():
    return pd.DataFrame(index=[
                                 'brute_force',
                                 'map',
                  ],
                  columns=['time', 'mem']).fillna('-')


# ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
print(f'start square')

DF = build_df()

TEST_DICT = _create_random_sparse_dict(0,10,(_rows,_columns),_sparsity,int)

RESULTS = tmb(
            ('brute_force', core_sparse_transpose_brute_force,[TEST_DICT],{}),
            ('map', core_sparse_transpose_map, [TEST_DICT], {}),
            rest_time=_rest_time,
            number_of_trials=_num_trials,
            verbose=0
)

DF.loc['brute_force', 'time'] = np.mean(RESULTS[0][0])
DF.loc['brute_force', 'mem'] = np.mean(RESULTS[1][0])
DF.loc['map', 'time'] = np.mean(RESULTS[0][1])
DF.loc['map', 'mem'] = np.mean(RESULTS[1][1])

print(DF)

print(f'end square')
# ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

# ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
print(f'start long')

DF = build_df()

TEST_DICT = _create_random_sparse_dict(0, 10, (int(_rows/2), int(_columns*2)),
                                       _sparsity, int)

RESULTS = tmb(
            ('brute_force', core_sparse_transpose_brute_force,[TEST_DICT],{}),
            ('map', core_sparse_transpose_map, [TEST_DICT], {}),
            rest_time=_rest_time,
            number_of_trials=_num_trials,
            verbose=0
)

DF.loc['brute_force', 'time'] = np.mean(RESULTS[0][0])
DF.loc['brute_force', 'mem'] = np.mean(RESULTS[1][0])
DF.loc['map', 'time'] = np.mean(RESULTS[0][1])
DF.loc['map', 'mem'] = np.mean(RESULTS[1][1])

print(DF)

print(f'end long')
# ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


# ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
print(f'start short')

DF = build_df()

TEST_DICT = _create_random_sparse_dict(0,10,(int(_rows*2), int(_columns/2)),
                                       _sparsity,int)


RESULTS = tmb(
            ('brute_force', core_sparse_transpose_brute_force,[TEST_DICT],{}),
            ('map', core_sparse_transpose_map, [TEST_DICT], {}),
            rest_time=_rest_time,
            number_of_trials=_num_trials,
            verbose=0
)

DF.loc['brute_force', 'time'] = np.mean(RESULTS[0][0])
DF.loc['brute_force', 'mem'] = np.mean(RESULTS[1][0])
DF.loc['map', 'time'] = np.mean(RESULTS[0][1])
DF.loc['map', 'mem'] = np.mean(RESULTS[1][1])

print(DF)

print(f'end short')
# ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

# 24_05_07
# start square
#                   time         mem
# brute_force   9.569878  328.666667
# map          21.724437  413.333333
# end square
# start long
#                   time        mem
# brute_force  10.138518  16.666667
# map          21.713491      106.0
# end long
# start short
#                   time  mem
# brute_force   7.957059  0.0
# map          19.733538  0.0
# end short



# 24_05_06
# start square
#                   time          mem
# brute_force   9.728309          0.0
# pandas       96.562638  1783.666667
# unzip_t_zip  79.122894   645.666667
# map          28.750932    41.333333
# in_place     91.602543          0.0
# end square

# start long
#                   time          mem
# brute_force  10.024974        126.0
# pandas       96.283246  1521.666667
# unzip_t_zip  80.941832   487.333333
# map          28.640498        157.0
# in_place     81.912707         17.0
# end long

# start short
#                   time         mem
# brute_force   8.969766         0.0
# pandas       94.848687      1331.0
# unzip_t_zip   78.64201  321.333333
# map          27.046026         0.0
# in_place     81.563347       -24.0
# end short















