# Authors:
#
#       Bill Sousa
#
# License: BSD 3 clause



import numpy as np
from pybear.new_numpy._random import Sparse
from pybear.utils._benchmarking import time_memory_benchmark as tmb
from pybear.utils._array_sparsity import array_sparsity




# OF THE FOUR METHODS:
# 1) choice
# 2) filter
# 3) serialized
# 4) iterative
# choice & filter ARE THE FASTEST, iterative IS NEXT FASTEST & HAS PERFECT
# ACCURACY, AND serialized IS BY FAR THE SLOWEST BUT ALSO HAS 100% ACCURACY.
# AT LOWER SIZES, filter APPEARS TO HAVE MARGINALLY BETTER ACCURACY THAN
# choice.






for _shape in [(500, 1000), (707,707), (1000,500)]:

    _min = -10
    _max = 10
    _sparsity = 10
    _dtype = np.int32

    print(f'\nRUNNING ({_shape[0]}, {_shape[1]}), sparsity = {_sparsity}%...')

    tmb(
        (
        '_choice',
        Sparse(_min, _max, _shape, _sparsity, dtype=_dtype, engine="choice").fit_transform,
        [],
        {}
        ),

        (
        '_filter',
        Sparse(_min, _max, _shape, _sparsity, dtype=_dtype, engine="filter").fit_transform,
        [],
        {}
        ),

        (
        '_serialized',
        Sparse(_min, _max, _shape, _sparsity, dtype=_dtype, engine="serialized").fit_transform,
        [],
        {}
        ),

        (
        '_iterative',
        Sparse(_min, _max, _shape, _sparsity, dtype=_dtype, engine="iterative").fit_transform,
        [],
        {}
        ),

        number_of_trials=1,
        rest_time=0.1,
        verbose=1
    )







# PRECISION TESTS
print(f'\nRunning precision tests...')
_min = 1
_max = 10
_dtype = np.uint8
SPARSITIES = (0, 10, 50, 90, 100)
SHAPES = ((100,100), (500,500), (1000,1000))#, (2000,2000))
SPARSITY_HOLDER = np.empty((4, len(SPARSITIES), len(SHAPES)))
for _sparsity_idx, _sparsity in enumerate(SPARSITIES):
    for _shape_idx, _shape in enumerate(SHAPES):
        # '_choice',
        a = Sparse(_min, _max, _shape, _sparsity, dtype=_dtype, engine='choice').fit_transform()
        SPARSITY_HOLDER[0, _sparsity_idx, _shape_idx] = array_sparsity(a)

        # '_filter',
        b = Sparse(_min, _max, _shape, _sparsity, dtype=_dtype, engine='filter').fit_transform()
        SPARSITY_HOLDER[1, _sparsity_idx, _shape_idx] = array_sparsity(b)

        # '_serialized',
        c = Sparse(_min, _max, _shape, _sparsity, dtype=_dtype, engine='serialized').fit_transform()
        SPARSITY_HOLDER[2, _sparsity_idx, _shape_idx] = array_sparsity(c)

        # '_iterative',
        d = Sparse(_min, _max, _shape, _sparsity, dtype=_dtype, engine='iterative').fit_transform()
        SPARSITY_HOLDER[3, _sparsity_idx, _shape_idx] = array_sparsity(d)

        del a, b, c, d


import pandas as pd
for engine_type in range(4):
    print(f"{['choice', 'filter', 'serialized', 'iterative'][engine_type]}:")
    print(pd.DataFrame(data=SPARSITY_HOLDER[engine_type, :, :].transpose(),
                       columns=SPARSITIES, index=SHAPES))
    print()





