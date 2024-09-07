import time
import numpy as np
import pandas as pd
from pybear.sparse_dict import sparse_dict as sd


TEST1, TEST2, TEST3, TEST4, TEST5 = [], [], [], [], []
n_methods = 5
for __ in range(50):
    print(f'trial {__ + 1}')
    if __ % n_methods == 0:
        t0 = time.time()
        OBJECT = {_: __ for _, __ in enumerate(range(1, 1000001))}
        TEST1.append(time.time() - t0)
    #         print(f'enumerate range elapsed time = {time.time() - t0}')
    elif __ % n_methods == 1:
        t0 = time.time()
        OBJECT = {_: __ for _, __ in enumerate(np.arange(1, 1000001))}
        #         print(f'enumerate np arange elapsed time = {time.time() - t0}')
        TEST2.append(time.time() - t0)
    elif __ % n_methods == 2:
        t0 = time.time()
        OBJECT = sd.create_random_py_int(1, 1000001, (1,1000001), 50)
        #         print(f'sparse dict create_random_50 elapsed time = {time.time() - t0}')
        TEST3.append(time.time() - t0)
    elif __ % n_methods == 3:
        t0 = time.time()
        OBJECT = {_: __ for _, __ in enumerate([x for x in range(1, 1000001)])}
        #         print(f'enumerate list comp elapsed time = {time.time() - t0}')
        TEST4.append(time.time() - t0)
    elif __ % n_methods == 4:
        t0 = time.time()
        SPARSE = np.random.randint(0, 2, (1, 1000001))[0]
        NON_ZEROS = np.nonzero(SPARSE)[0]
        RANDS = np.random.randint(0, 10, (1, len(NON_ZEROS)))[0]
        OBJECT = {NON_ZEROS[_]: RANDS[_] for _ in range(len(NON_ZEROS))}
        #         print(f'np non-zero sparse dict elapsed time = {time.time() - t0}')
        TEST5.append(time.time() - t0)
        del SPARSE, NON_ZEROS, RANDS

    del OBJECT


print()
print(f'average enumerate range elapsed time = {np.average(TEST1)}')
print(f'average enumerate np arange elapsed time = {np.average(TEST2)}')
print(f'average sparse dict create_random_50 elapsed time = {np.average(TEST3)}')
print(f'average enumerate list comp elapsed time = {np.average(TEST4)}')
print(f'average np non-zero sparse dict elapsed time = {np.average(TEST5)}')


del TEST1, TEST2, TEST3, TEST4, TEST5










