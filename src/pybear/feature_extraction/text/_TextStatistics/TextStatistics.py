# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Sequence, Self

from copy import deepcopy

import numpy as np



class TextStatistics:


    def __init__(self) -> None:

        """
        pizza finalize this.
        Print statistics about a list of words to the screen. Returns
        nothing. Statistics reported include
        - size
        - uniques count
        - average length and standard deviation
        - max word length
        - min word length
        - 'starts with' frequency
        - letter frequency
        - top word frequencies
        - top longest words


        Parameters
        ----------
        WORDS:
            Sequence[str] - a single list-like vector of words to report
            statistics for. Words do not need to be in the Lexicon.


    Return
    ------
    -
        None

    """


    def partial_fit(self, WORDS: Sequence[str]) -> Self:
        pass


    _lp = 5  # left pad
    _rp = 15  # right pad


    def _printer(_description:str, value:any) -> None:
        nonlocal _lp, _rp
        print(f' ' * _lp + str(_description).ljust(2*_rp), value)


    _size = len(WORDS)
    _LENS = np.fromiter(map(len, WORDS), dtype=np.uint8)
    _UNIQUES, _COUNTS = np.unique(WORDS, return_counts=True)

    print(f'\nSTATISTICS:')
    _printer(f'Size:', _size)
    _printer(f'Uniques count:', len(_UNIQUES))
    _printer(f'Average length', sum(np.fromiter(_LENS, dtype=np.int8)) / _size)
    _printer(f'Std deviation', np.std(np.fromiter(_LENS, dtype=np.int8)))
    _printer(f'Max len:', max(_LENS))
    _printer(f'Min len:', min(_LENS))

    del _LENS, _printer

    # dictionary for holding frequency counts of letters
    LETTER_DICT = {
        'a':0, 'b':0, 'c':0, 'd':0, 'e':0, 'f':0, 'g':0, 'h':0, 'i':0,
        'j':0, 'k':0, 'l':0, 'm':0, 'n':0, 'o':0, 'p':0, 'q':0, 'r':0,
        's':0, 't':0, 'u':0, 'v':0, 'w':0, 'x':0, 'y':0, 'z':0, 'other':0
    }

    # "STARTS WITH" FREQ ** * ** * ** * ** * ** * ** * ** * ** * ** * **
    START_DICT = deepcopy(LETTER_DICT)
    for word in WORDS:
        try:
            START_DICT[word[0].lower()] += 1
        except:
            START_DICT['other'] += 1

    KEYS = np.fromiter(START_DICT.keys(), dtype='<U5')
    # DONT USE np.int8 OR 16! NUMBERS TOO BIG!
    VALUES = np.fromiter(START_DICT.values(), dtype=np.int32)
    MASK = np.flip(np.argsort(VALUES))
    SORTED_DICT = {k:v for k,v in zip(KEYS[MASK], VALUES[MASK])}
    SORTED_KEYS = np.fromiter(SORTED_DICT.keys(), dtype='<U5')

    # CHANGE KEYS FOR EASY PRINT
    for new_key in range(26):
        START_DICT[new_key] = START_DICT.pop(KEYS[new_key])
        SORTED_DICT[new_key] = SORTED_DICT.pop(SORTED_KEYS[new_key])

    del VALUES, MASK


    print(f'\n"STARTS WITH" FREQUENCY:')

    for i in range(26):
        print(_lp*' ' + f'{KEYS[i].upper()}:'.ljust(_rp), end='')
        print(f'{START_DICT[i]}'.ljust(2*_rp), end='')
        print(_lp*' ' + f'{SORTED_KEYS[i].upper()}:'.ljust(_rp), end='')
        print(f'{SORTED_DICT[i]}')

    del START_DICT, KEYS, SORTED_DICT, SORTED_KEYS
    # END "STARTS WITH" FREQ ** * ** * ** * ** * ** * ** * ** * ** * ** *


    # LETTER FREQ ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
    FREQ_DICT = deepcopy(LETTER_DICT)
    for word in WORDS:
        for letter in word:
            try: FREQ_DICT[letter.lower()] += 1
            except: FREQ_DICT['other'] += 1

    KEYS = np.fromiter(FREQ_DICT.keys(), dtype='<U5')
    # DONT USE np.int8 OR 16! NUMBERS TOO BIG!
    VALUES = np.fromiter(FREQ_DICT.values(), dtype=np.int32)
    MASK = np.flip(np.argsort(VALUES))
    SORTED_DICT = {k:v for k,v in zip(KEYS[MASK], VALUES[MASK])}
    SORTED_KEYS = np.fromiter(SORTED_DICT.keys(), dtype='<U5')

    # CHANGE KEYS FOR EASY PRINT
    for new_key in range(27):
        FREQ_DICT[new_key] = FREQ_DICT.pop(KEYS[new_key])
        SORTED_DICT[new_key] = SORTED_DICT.pop(SORTED_KEYS[new_key])

    del VALUES, MASK

    print(f'\nOVERALL LETTER FREQUENCY:')

    for i in range(26):
        print(_lp*' ' + f'{KEYS[i].upper()}:'.ljust(_rp), end='')
        print(f'{FREQ_DICT[i]}'.ljust(2*_rp), end='')
        print(_lp*' ' + f'{SORTED_KEYS[i].upper()}:'.ljust(_rp), end='')
        print(f'{SORTED_DICT[i]}')

    del FREQ_DICT, KEYS, SORTED_DICT, SORTED_KEYS
    # END LETTER FREQ ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *



    # TOP WORD FREQUENCY ** * ** * ** * ** * ** * ** * ** * ** * ** * **
    n = min(20, len(_UNIQUES))
    print(f'\n TOP {n} WORD FREQUENCY:')
    MASK = np.flip(np.argsort(_COUNTS))[:n]

    print(_lp*' ' + (f'WORD').ljust(2*_rp) + f'FREQUENCY')
    for i in range(n):
        print(_lp * ' ' + f'{_UNIQUES[..., MASK][i]}'.ljust(2*_rp), end='')
        print(f'{_COUNTS[..., MASK][i]}')

    # END TOP WORD FREQUENCY ** * ** * ** * ** * ** * ** * ** * ** * ** *

    # TOP LONGEST WORDS ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
    n = min(20, len(_UNIQUES))
    print(f'\nTOP {n} LONGEST WORDS:')

    _LENS = np.fromiter(map(len, _UNIQUES), dtype=np.int8)

    MASK = np.flip(np.argsort(_LENS))
    LONGEST_WORDS = _UNIQUES[MASK][:n]
    _LENS = _LENS[MASK][:n]
    del MASK

    print(_lp*' ' + f'WORD'.ljust(3*_rp) + f'LENGTH')
    for i in range(n):
        print(_lp*' ' + f'{(LONGEST_WORDS[i])}'.ljust(3*_rp), end='')
        print(f'{_LENS[i]}')

    del LONGEST_WORDS
    # END TOP LONGEST WORDS ** * ** * ** * ** * ** * ** * ** * ** * ** *

    del _UNIQUES, _COUNTS, _LENS, _lp, _rp








