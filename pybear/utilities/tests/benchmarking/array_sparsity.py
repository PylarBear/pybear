# Authors:
#
#       Bill Sousa
#
# License: BSD 3 clause




from utilities._array_sparsity import array_sparsity
from utilities._benchmarking import time_memory_benchmark as tmb
import numpy as np




_min = 1
_max = 10
_rows = 10_000
_cols = 10_000
_sparsity = 90

a = np.random.randint(_min, _max, (_rows, _cols), dtype=np.uint8)


tmb(
    ('np.random.randint',
     np.random.randint,
     [_min, _max, (_rows, _cols)],
     {'dtype': np.uint8}
    ),
    ('pybear.array_sparsity', array_sparsity, [a], {}),
    number_of_trials=10,
    rest_time=1,
    verbose=1
)












