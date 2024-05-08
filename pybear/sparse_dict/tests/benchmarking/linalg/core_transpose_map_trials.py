# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

import numpy as np
import pandas as pd
from pybear.utils import time_memory_benchmark as tmb
from pybear.sparse_dict._random_ import _create_random_sparse_dict
from pybear.sparse_dict import _validation as val
from pybear.sparse_dict._utils import  outer_len, inner_len_quick


def core_sparse_transpose_map_duality(DICT1:dict) -> dict:
    """Transpose a sparse dict to a sparse dict using map tricks"""

    if val._is_sparse_inner(DICT1):
        DICT1 = {0: DICT1}

    old_outer_len = outer_len(DICT1)
    old_inner_len = inner_len_quick(DICT1)

    def placeholder(x):
        NEW_DICT[int(x)][int(old_outer_len - 1)] = \
            NEW_DICT[x].get(old_outer_len - 1, 0)


    def appender(x, outer_key):
        NEW_DICT[int(x)][int(outer_key)] = DICT1[outer_key][x]


    if old_inner_len >= old_outer_len:

        # CREATE TRANSPOSED DICT & FILL WITH {outer_keys:{}}
        NEW_DICT = dict((map(lambda x: (int(x), {}), range(old_inner_len))))

        # POPULATE TRANSPOSED DICT WITH VALUES
        list(map(lambda outer_key: list(
            map(lambda x: appender(x, outer_key), DICT1[outer_key])),
        DICT1))

        # REMOVE TRANSPOSED PLACEHOLDERS
        list(map(lambda x: NEW_DICT[old_inner_len-1].pop(x),
             [k for k, v in NEW_DICT[old_inner_len-1].items() if v == 0]))

    elif old_outer_len > old_inner_len:

        NEW_DICT = {}

        for old_inner_idx in range(old_inner_len):
            HOLDER = dict((
                zip(range(old_outer_len),
                    list(map(lambda x: DICT1[x].get(old_inner_idx, 0), DICT1))
                )
            ))
            list(map(lambda x: HOLDER.pop(x) if HOLDER[x]==0 else 1, list(HOLDER.keys())))
            NEW_DICT[int(old_inner_idx)] = HOLDER

    # PLACEHOLDERS
    list(map(lambda x: placeholder(x), NEW_DICT))

    del placeholder, appender

    return NEW_DICT



def core_sparse_transpose_map_no_duality_1(DICT1:dict) -> dict:
    """Transpose a sparse dict to a sparse dict using map tricks"""

    if val._is_sparse_inner(DICT1):
        DICT1 = {0: DICT1}

    old_outer_len = outer_len(DICT1)
    old_inner_len = inner_len_quick(DICT1)

    def placeholder(x):
        NEW_DICT[int(x)][int(old_outer_len - 1)] = \
            NEW_DICT[x].get(old_outer_len - 1, 0)


    def appender(x, outer_key):
        NEW_DICT[int(x)][int(outer_key)] = DICT1[outer_key][x]


    # CREATE TRANSPOSED DICT & FILL WITH {outer_keys:{}}
    NEW_DICT = dict((map(lambda x: (int(x), {}), range(old_inner_len))))

    # POPULATE TRANSPOSED DICT WITH VALUES
    list(map(lambda outer_key: list(
        map(lambda x: appender(x, outer_key), DICT1[outer_key])), DICT1))

    # REMOVE TRANSPOSED PLACEHOLDERS
    list(map(lambda x: NEW_DICT[old_inner_len-1].pop(x),
             [k for k,v in NEW_DICT[old_inner_len-1].items() if v==0]
    ))

    # PLACEHOLDERS
    list(map(lambda x: placeholder(x), NEW_DICT))

    del placeholder

    return NEW_DICT



def core_sparse_transpose_map_no_duality_2(DICT1:dict) -> dict:
    """Transpose a sparse dict to a sparse dict using map tricks"""

    if val._is_sparse_inner(DICT1):
        DICT1 = {0: DICT1}

    old_outer_len = outer_len(DICT1)
    old_inner_len = inner_len_quick(DICT1)

    def placeholder(x):
        NEW_DICT[int(x)][int(old_outer_len - 1)] = \
            NEW_DICT[x].get(old_outer_len - 1, 0)

    NEW_DICT = {}

    for old_inner_idx in range(old_inner_len):
        HOLDER = dict((
            zip(range(old_outer_len),
                list(map(lambda x: DICT1[x].get(old_inner_idx, 0), DICT1))
            )
        ))
        list(map(lambda x: HOLDER.pop(x) if HOLDER[x]==0 else 1, list(HOLDER.keys())))
        NEW_DICT[int(old_inner_idx)] = HOLDER

    list(map(lambda x: placeholder(x), NEW_DICT))

    del placeholder

    return NEW_DICT







if __name__ == '__main__':

    # LINUX TRIALS 24_05_06
    # _rows = 6_000
    # _columns = 6_000
    # _sparsity = 50
    # _rest_time = 2
    # _num_trials = 5

    # start square
    #                   time     mem
    # duality      40.254596  1005.0
    # no_duality1  39.873809  1008.0
    # no_duality2  66.085016  2140.0
    # end square
    # start long
    #                   time          mem
    # duality       39.28792   219.666667
    # no_duality1   39.22029   199.666667
    # no_duality2  60.263112  1315.666667
    # end long
    # start short
    #                   time          mem
    # duality      59.687735  1111.666667
    # no_duality1  36.590809          0.0
    # no_duality2  59.546833  1111.666667
    # end short


    def build_df():
        return pd.DataFrame(index=['duality', 'no_duality1', 'no_duality2'],
                          columns=['time', 'mem']).fillna('-')

    _rows = 6_000
    _columns = 6_000
    _sparsity = 50
    _rest_time = 2
    _num_trials = 5

    # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
    print(f'start square')

    DF = build_df()

    TEST_DICT = _create_random_sparse_dict(0,10,(_rows,_columns),_sparsity,int)


    RESULTS = tmb(
        ('duality', core_sparse_transpose_map_duality,[TEST_DICT],{}),
        ('no_duality1', core_sparse_transpose_map_no_duality_1, [TEST_DICT], {}),
        ('no_duality2', core_sparse_transpose_map_no_duality_2, [TEST_DICT], {}),
        rest_time=_rest_time,
        number_of_trials=_num_trials,
        verbose=0
    )

    DF.loc['duality', 'time'] = np.mean(RESULTS[0][0])
    DF.loc['duality', 'mem'] = np.mean(RESULTS[1][0])
    DF.loc['no_duality1', 'time'] = np.mean(RESULTS[0][1])
    DF.loc['no_duality1', 'mem'] = np.mean(RESULTS[1][1])
    DF.loc['no_duality2', 'time'] = np.mean(RESULTS[0][2])
    DF.loc['no_duality2', 'mem'] = np.mean(RESULTS[1][2])

    print(DF)

    print(f'end square')
    # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
    print(f'start long')

    DF = build_df()

    TEST_DICT = _create_random_sparse_dict(0, 10, ((int(_rows/2)),
                                           int(2*_columns)), _sparsity, int)

    RESULTS = tmb(
        ('duality', core_sparse_transpose_map_duality, [TEST_DICT], {}),
        ('no_duality1', core_sparse_transpose_map_no_duality_1, [TEST_DICT], {}),
        ('no_duality2', core_sparse_transpose_map_no_duality_2, [TEST_DICT], {}),
        rest_time=_rest_time,
        number_of_trials=_num_trials,
        verbose=0
    )

    DF.loc['duality', 'time'] = np.mean(RESULTS[0][0])
    DF.loc['duality', 'mem'] = np.mean(RESULTS[1][0])
    DF.loc['no_duality1', 'time'] = np.mean(RESULTS[0][1])
    DF.loc['no_duality1', 'mem'] = np.mean(RESULTS[1][1])
    DF.loc['no_duality2', 'time'] = np.mean(RESULTS[0][2])
    DF.loc['no_duality2', 'mem'] = np.mean(RESULTS[1][2])

    print(DF)

    print(f'end long')
    # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
    print(f'start short')

    DF = build_df()

    TEST_DICT = _create_random_sparse_dict(0,10,((int(2*_rows)),
                                             int(_columns/2)),_sparsity,int)


    RESULTS = tmb(
        ('duality', core_sparse_transpose_map_duality,[TEST_DICT],{}),
        ('no_duality1', core_sparse_transpose_map_no_duality_1, [TEST_DICT], {}),
        ('no_duality2', core_sparse_transpose_map_no_duality_2, [TEST_DICT], {}),
        rest_time=_rest_time,
        number_of_trials=_num_trials,
        verbose=0
    )

    DF.loc['duality', 'time'] = np.mean(RESULTS[0][0])
    DF.loc['duality', 'mem'] = np.mean(RESULTS[1][0])
    DF.loc['no_duality1', 'time'] = np.mean(RESULTS[0][1])
    DF.loc['no_duality1', 'mem'] = np.mean(RESULTS[1][1])
    DF.loc['no_duality2', 'time'] = np.mean(RESULTS[0][2])
    DF.loc['no_duality2', 'mem'] = np.mean(RESULTS[1][2])

    print(DF)

    print(f'end short')
    # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

























