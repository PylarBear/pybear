# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import os


def _read_dummy_strings():

    module_dir = os.path.dirname(os.path.abspath(__file__))
    file = '_dummy_strings'

    WORDS = []
    with open(os.path.join(module_dir, file)) as f:
        for line in f:
            if line == '\n':
                continue
            WORDS.append(line.replace('\n', ''))


    return WORDS








