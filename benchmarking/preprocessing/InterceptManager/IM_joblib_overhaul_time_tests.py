# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import time

import numpy as np

from pybear.preprocessing._InterceptManager.InterceptManager import \
    InterceptManager as IM



X = np.random.randint(0, 10, (3352, 20_000))


trfm = IM(
    keep = 'last',
    equal_nan = True,
    rtol = 1e-5,
    atol = 1e-8,
    n_jobs = -1
)
times = []
for i in range(10):
    print(f'BENCHMARK start fit')
    t0 = time.perf_counter()
    trfm.fit(X)
    tf = time.perf_counter() - t0
    print(f'BENCHMARK end fit')
    print(f'fit time = {tf: ,.3f}')
    times.append(tf)
# print(f'BENCHMARK start transform')
# out = trfm.transform(X)
# print(f'BENCHMARK end transform')
print(f'average times = {float(np.mean(times)): ,.3f}')

# with joblib:
# 8.701 sec

# for loop:
# 9.703 sec

# chunk for loop 200 columns:
# 6.326 sec


