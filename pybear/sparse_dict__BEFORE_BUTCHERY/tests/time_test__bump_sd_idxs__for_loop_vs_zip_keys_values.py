import numpy as np, sparse_dict as sd, pandas as pd
from debug import time_memory_tester as tmt



# 12/25/22 bump_keys_zip_keys_values WINS HANDS DOWN
#          size1         size2 for_time zip_time for_mem zip_mem
# 0  (3000, 500)   (3000, 500)   26.481    0.023   0.000   0.000
# 1  (3000, 500)   (5000, 500)   44.049    0.027   0.000   0.000
# 2  (3000, 500)  (10000, 500)   93.195    0.025   0.000   0.000


def for_loop(DICT1, DICT2):

    NEW_DICT = {}

    for dumidx in range(len(DICT2) - 1, -1, -1):  # SHOULD BE A CLEAN DICT, OUTER IDXS SEQUENTIAL FROM 0
        NEW_DICT[int(dumidx + sd.outer_len(DICT1))] = DICT2[dumidx]

    for idx in sorted(list(NEW_DICT.keys())):  # REORDER
        NEW_DICT[int(idx)] = NEW_DICT.pop(idx)

    NEW_DICT = DICT1 | NEW_DICT

    return NEW_DICT



def zip_keys_values(DICT1, DICT2):
    NEW_KEYS = np.fromiter(DICT2.keys(), dtype=np.int16) + sd.outer_len(DICT1)

    return DICT1 | dict((zip(NEW_KEYS.tolist(), DICT2.values())))


# CONGRUENCY TEST
def congruency_test():
    for _size1 in ((300,500),(500,500),(1000,500)):
        for _size2 in ((300, 500), (500, 500), (1000, 500)):
            DICT1 = sd.create_random_py_int(1, 10, _size1, _sparsity=90)
            DICT2 = sd.create_random_py_int(1, 10, _size2, _sparsity=90)
            _ = for_loop(DICT1, DICT2)
            __ = zip_keys_values(DICT1, DICT2)
            if not sd.safe_sparse_equiv(_,__):
                raise Exception(f'DISASTER')
    print(f'\033[92m*** ALL CONGRUENCY TESTS PASSED ***\033[0m\x1B[0m')

# congruency_test()
# quit()

RESULTS = pd.DataFrame(index=range(9), columns=['size1','size2','for_time','zip_time','for_mem','zip_mem']).fillna('-')

ctr=-1
for _size1 in ((3000,500),(5000,500),(10000,500)):
    for _size2 in ((3000, 500), (5000, 500), (10000, 500)):
        ctr+=1
        DICT1 = sd.create_random_py_int(1, 10, _size1, _sparsity=90)
        DICT2 = sd.create_random_py_int(1, 10, _size2, _sparsity=90)

        TIME_MEM = \
            tmt.time_memory_tester(
                                        ('for_loop', for_loop, [DICT1, DICT2], {}),
                                        ('zip_keys_values', zip_keys_values, [DICT1, DICT2], {}),
                                         number_of_trials=5,
                                         rest_time=1
            )

        for col_idx, value in enumerate((_size1, _size2, *TIME_MEM[0].mean(axis=1).ravel(), *TIME_MEM[1].mean(axis=1).ravel())):
            RESULTS[['size1', 'size2', 'for_time', 'zip_time', 'for_mem', 'zip_mem'][col_idx]][ctr] = value

        print(RESULTS)
print(f'\033[92m*** TIME / MEM TESTS DONE ***\033[0m\x1B[0m')























