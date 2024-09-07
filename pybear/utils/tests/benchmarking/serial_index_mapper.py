# Authors:
#
#       Bill Sousa
#
# License: BSD 3 clause


import numpy as np
from pybear.utils._serial_index_mapper import serial_index_mapper as sim
from pybear.utils._benchmarking import time_memory_benchmark as tmb




_x = 200
_y = 200
_z = 200

tmb(
    ('serial_index_mapper',
        sim,
        [(_x,_y,_z),
        (12342, 3438595)],
        {'n_jobs':1}
    ),
    number_of_trials=4,
    rest_time=1,
    verbose=1
)


_v = 10
_w = 10
_x = 10
_y = 10
_z = 10

tmb(
    ('serial_index_mapper',
        sim,
        [(_v, _w, _x,_y,_z),
         (12342, 34385)],
        {'n_jobs':1}
    ),
    number_of_trials=4,
    rest_time=1,
    verbose=1
)



_x = 200
_y = 200
_z = 200

# 24_04_11_16_04_00
# THREADS
# serial_index_mapper     time = 18.402 +/- 0.318 sec; mem = 13.000 +/- 1.000 MB

# PROCESSES
# serial_index_mapper     time = 5.340 +/- 0.189 sec; mem = 3.500 +/- 0.500 MB


tmb(
    ('serial_index_mapper',
        sim,
        [(_x,_y,_z), np.arange(np.prod((_x, _y, _z)))[:100_000]],
        {'n_jobs': -1}
    ),
    number_of_trials=4,
    rest_time=1,
    verbose=1
)







