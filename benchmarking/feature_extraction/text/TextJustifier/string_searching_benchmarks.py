# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


# compare the time and memory usage of
# map(lambda x: random_str.find(x), seps)
# and
# map(str.find, (random_str for _ in seps), seps)


# LINUX, len(STRING)=1e9, n_seps=5
# function_map_lambda     time = 0.478 +/- 0.013 sec; mem = 0.000 +/- 0.000 MB
# function_map_find       time = 0.469 +/- 0.003 sec; mem = 0.000 +/- 0.000 MB



import random
import string

from pybear.utilities import time_memory_benchmark as tmb





def generate_random_str(length: int) -> str:
    _pool = string.ascii_letters
    return ''.join(random.choice(_pool) for _ in range(length))

# make a really long string
random_str = generate_random_str(10_000_000)
# make it even longer
random_str *= 100
# put the seps at the end of the string so the functions have to search all of it
random_str += '*(&)@'

seps = {'*', '(', '&', ')', '@'}



def function_map_lambda(random_str, seps):
    return list(map(lambda x: (random_str.find(x), x), seps))


def function_map_find(random_str, seps):
    return list(map(str.find, (random_str for _ in seps), seps))




tmb(
    ('function_map_lambda', function_map_lambda, [random_str, seps], {}),
    ('function_map_find', function_map_find, [random_str, seps], {}),
    rest_time=1,
    number_of_trials=5
)
















