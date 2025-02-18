# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.feature_extraction.text._Lexicon._methods._find_duplicates import \
    _find_duplicates



if __name__ == '__main__':

    # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
    print(f'NO DUPLICATES: \n')
    string_frequency_ = {'an':1, 'apple': 1, 'a':1, 'day': 1}
    out = _find_duplicates(string_frequency_)
    # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
    print(f'HAS DUPLICATES: \n')
    string_frequency_ = {'an':2, 'apple': 2, 'a':1, 'day': 1}
    out = _find_duplicates(string_frequency_)
    # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
    print(f'HAS DUPLICATES: \n')
    string_frequency_ = {'an':1, 'apple': 1, 'a': 2, 'day': 2}
    out = _find_duplicates(string_frequency_)
    # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


