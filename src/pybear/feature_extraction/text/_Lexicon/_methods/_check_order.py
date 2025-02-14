# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import numpy.typing as npt

import numbers

import numpy as np


# pizza this needs test


def _check_order(
    lexicon_: npt.NDArray[str]
) -> npt.NDArray[str]:

    """
    Determine if words in the Lexicon files are out of alphabetical
    order by comparing the words as stored against a sorted vector of
    the words. Displays any out-of-order words to screen and return a
    numpy vector of the words.


    Parameters
    ----------
    lexicon_:
        NDArray[str] - the numpy vector containing the pybear lexicon.


    Return
    ------
    -
        OUT_OF_ORDER: NDArray[str] - vector of any out of sequence
        words in the lexicon.


    """


    assert isinstance(lexicon_, np.ndarray)


    # np.unique sorts asc alpha
    __ = np.unique(lexicon_)

    if np.array_equiv(lexicon_, __):

        print(f'\n*** LEXICON IS IN ALPHABETICAL ORDER ***\n')

        return np.empty(0, dtype='<U40')
    elif len(lexicon_) != len(__):

        print(f'\n*** LEXICON HAS DUPLICATE ENTRIES ***\n')

    else:
        OUT_OF_ORDER = []
        for idx in range(len(__)):
            # len(__) must <= len(lexicon_)
            if lexicon_[idx] != __[idx]:
                OUT_OF_ORDER.append(__[idx])
        if len(OUT_OF_ORDER) > 0:
            print(f'OUT OF ORDER:')
            print(OUT_OF_ORDER)

        return np.array(OUT_OF_ORDER, dtype='<U40')







