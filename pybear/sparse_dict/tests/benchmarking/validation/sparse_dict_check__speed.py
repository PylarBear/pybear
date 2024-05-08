# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

from pybear import sparse_dict
from pybear.sparse_dict._validation import _sparse_dict_check
from pybear.utils._benchmarking import time_memory_benchmark as tmb


def build_sparse_dict(_shape):
    return sparse_dict.random.randint(0,10,_shape,50,dtype=int)


SD1 = build_sparse_dict((5000,5000))
SD2 = build_sparse_dict((1,25_000_000))
SD3 = build_sparse_dict((25_000, 1_000))

RESULTS = tmb(
        ('sdc', _sparse_dict_check, [SD1], {}),
        ('sdc', _sparse_dict_check, [SD2], {}),
        ('sdc', _sparse_dict_check, [SD3], {}),
        rest_time=3,
        number_of_trials=5,
        verbose=1
)

# FIRST BUILD PRE 24_05_06
# sdc     time = 13.059 +/- 0.076 sec; mem = -32.333 +/- 45.726 MB
# sdc     time = 12.822 +/- 0.021 sec; mem = 0.000 +/- 0.000 MB
# sdc     time = 13.323 +/- 0.021 sec; mem = 65.333 +/- 45.492 MB

# CURRENT BUILD 24_05_06
# sdc     time = 8.107 +/- 0.154 sec; mem = 0.000 +/- 0.000 MB
# sdc     time = 8.068 +/- 0.189 sec; mem = 0.000 +/- 0.000 MB
# sdc     time = 8.428 +/- 0.069 sec; mem = 1.333 +/- 1.247 MB





