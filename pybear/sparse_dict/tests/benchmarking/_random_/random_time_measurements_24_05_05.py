# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

import numpy as np
import pandas as pd
from pybear.utils import time_memory_benchmark as tmb
from pybear.sparse_dict._random_ import (
                                            randint,
                                            uniform
)


# PEFORMANCE BENCHMARK ON LINUX 24_05_05

#                randint_time randint_mem uniform_time uniform_mem
# (2000, 2000)       0.453575        32.5     0.516249       4.375
# (4000, 4000)       1.549447     108.625     1.736918     132.875
# (6000, 6000)       3.096894     205.875     3.515886     201.875
# (8000, 8000)       5.402555     551.625     6.073368     551.875
# (10000, 10000)     8.037568      670.25     9.070514       670.0


_min = 2_000
_max = 10_001
_itvl= 2_000

DF = pd.DataFrame(index=[f"{(i, i)}" for i in range(_min, _max, _itvl)],
                  columns=['randint_time', 'randint_mem', 'uniform_time',
                           'uniform_mem']).fillna('-')

ctr = 0
for _dim in range(_min, _max, _itvl):

    RESULTS = tmb(
            ('randint', randint, (0, 10, (_dim, _dim), 90), {'dtype':np.uint8}),
            ('uniform', uniform, (0, 10, (_dim, _dim), 90), {}),
            rest_time=1,
            number_of_trials=10
    )

    DF.iloc[ctr, 0] = np.mean(RESULTS[0][0])
    DF.iloc[ctr, 2] = np.mean(RESULTS[0][1])
    DF.iloc[ctr, 1] = np.mean(RESULTS[1][0])
    DF.iloc[ctr, 3] = np.mean(RESULTS[1][1])

    ctr += 1


print(f"results:")
print(DF)
















