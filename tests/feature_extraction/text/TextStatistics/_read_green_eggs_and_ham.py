# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import os


def _read_green_eggs_and_ham():

    module_dir = os.path.dirname(os.path.abspath(__file__))
    file = '_green_eggs_and_ham'

    WORDS = []
    with open(os.path.join(module_dir, file)) as f:
        for line in f:
            if line == '\n':
                continue
            WORDS.append(line.replace('\n', ''))


    return WORDS








