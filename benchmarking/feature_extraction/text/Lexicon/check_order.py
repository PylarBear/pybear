# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.feature_extraction.text._Lexicon._methods._check_order import \
    _check_order



if __name__ == '__main__':

    # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
    print(f'IN ASC ALPHABETICAL: \n')
    lexicon_ = ['a', 'an', 'apple', 'day']
    out = _check_order(lexicon_)
    # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
    print(f'HAS DUPLICATES: \n')
    lexicon_ = ['an', 'an', 'apple', 'day']
    out = _check_order(lexicon_)

    # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
    print(f'OUT OF ORDER: \n')
    lexicon_ = ['an', 'a', 'apple', 'day']
    out = _check_order(lexicon_)
    # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *






