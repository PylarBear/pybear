import numpy as np
from copy import deepcopy


def statistics(WORDS_AS_LIST):
    _size = len(WORDS_AS_LIST)
    LENS = np.fromiter(map(len, WORDS_AS_LIST), dtype=np.int8)

    _pad = lambda x: f' '*5 + str(x)
    __ = 15
    print(f'\nSTATISTICS:')
    print(_pad(f'Size:').ljust(2*__), _size)
    print(_pad(f'Uniques count:').ljust(2*__), len(np.unique(WORDS_AS_LIST)))
    print(_pad(f'Average length').ljust(2*__), sum(np.fromiter(LENS, dtype=np.int8)) / _size)
    print(_pad(f'Std deviation').ljust(2*__), np.std(np.fromiter(LENS, dtype=np.int8)))
    print(_pad(f'Max len:').ljust(2*__), max(LENS))
    print(_pad(f'Min len:').ljust(2*__), min(LENS))

    del _size, LENS


    LETTER_DICT = {'a':0, 'b':0, 'c':0, 'd':0, 'e':0, 'f':0, 'g':0, 'h':0, 'i':0, 'j':0, 'k':0, 'l':0, 'm':0,
                   'n':0, 'o':0, 'p':0, 'q':0, 'r':0, 's':0, 't':0, 'u':0, 'v':0, 'w':0, 'x':0, 'y':0, 'z':0,
                   'other':0}

    # "STARTS WITH" FREQ
    START_DICT = deepcopy(LETTER_DICT)
    for word in WORDS_AS_LIST:
        try: START_DICT[word[0].lower()] += 1
        except: START_DICT['other'] += 1

    KEYS = np.fromiter(START_DICT.keys(), dtype='<U5')
    VALUES = np.fromiter(START_DICT.values(), dtype=np.int32)   # DONT USE np.int8 OR 16! NUMBERS TOO BIG!
    MASK = np.flip(np.argsort(VALUES))
    SORTED_DICT = {k:v for k,v in zip(KEYS[MASK], VALUES[MASK])}
    SORTED_KEYS = np.fromiter(SORTED_DICT.keys(), dtype='<U5')

    # CHANGE KEYS FOR EASY PRINT
    for new_key in range(27):
        a = KEYS
        b = SORTED_KEYS
        START_DICT[new_key] = START_DICT[a[new_key]]; del START_DICT[a[new_key]]
        SORTED_DICT[new_key] = SORTED_DICT[b[new_key]]; del SORTED_DICT[b[new_key]]
    del a, b, VALUES, MASK


    print(f'\n"STARTS WITH" FREQUENCY:')
    a = START_DICT
    b = KEYS
    c = SORTED_DICT
    d = SORTED_KEYS

    [print(f'{_pad(b[i].upper())}:'.ljust(__) + f'{a[i]}'.ljust(2*__) + f'{_pad(d[i].upper())}:'.ljust(__) + f'{c[i]}') for i in range(27)]

    del START_DICT, KEYS, SORTED_DICT, SORTED_KEYS

    # LETTER FREQ
    FREQ_DICT = deepcopy(LETTER_DICT)
    for word in WORDS_AS_LIST:
        for letter in word:
            try: FREQ_DICT[letter.lower()] += 1
            except: FREQ_DICT['other'] += 1

    KEYS = np.fromiter(FREQ_DICT.keys(), dtype='<U5')
    VALUES = np.fromiter(FREQ_DICT.values(), dtype=np.int32)   # DONT USE np.int8 OR 16! NUMBERS TOO BIG!
    MASK = np.flip(np.argsort(VALUES))
    SORTED_DICT = {k:v for k,v in zip(KEYS[MASK], VALUES[MASK])}
    SORTED_KEYS = np.fromiter(SORTED_DICT.keys(), dtype='<U5')

    # CHANGE KEYS FOR EASY PRINT
    for new_key in range(27):
        a = KEYS
        b = SORTED_KEYS
        FREQ_DICT[new_key] = FREQ_DICT[a[new_key]]; del FREQ_DICT[a[new_key]]
        SORTED_DICT[new_key] = SORTED_DICT[b[new_key]]; del SORTED_DICT[b[new_key]]
    del a, b, VALUES, MASK

    print(f'\nOVERALL LETTER FREQUENCY:')
    a = FREQ_DICT
    b = KEYS
    c = SORTED_DICT
    d = SORTED_KEYS

    [print(f'{_pad(b[i].upper())}:'.ljust(__) + f'{a[i]}'.ljust(2*__) + f'{_pad(d[i]).upper()}:'.ljust(__) + f'{c[i]}') for i in range(27)]

    del FREQ_DICT, KEYS, SORTED_DICT, SORTED_KEYS


    # TOP WORD FREQUENCY
    n = 20
    print(f'\n TOP {n} WORD FREQUENCY:')
    UNIQUES, COUNTS = np.unique(WORDS_AS_LIST, return_counts=True)
    MASK = np.flip(np.argsort(COUNTS))[:n]

    print(_pad(f'WORD').ljust(2*__) + f'FREQUENCY')
    [print(f'{_pad(UNIQUES[..., MASK][i])}'.ljust(2*__) + f'{COUNTS[..., MASK][i]}') for i in range(n)]
    del UNIQUES, COUNTS

    # TOP LONGEST WORDS
    n = 20
    print(f'\nTOP {n} LONGEST WORDS:')

    UNIQUES = np.unique(WORDS_AS_LIST)
    LENS = np.fromiter(map(len, UNIQUES), dtype=np.int8)

    MASK = np.flip(np.argsort(LENS))
    LONGEST_WORDS = UNIQUES[MASK][:n]
    LENS = LENS[MASK][:n]
    del MASK, UNIQUES

    print(_pad(f'WORD').ljust(3*__) + f'LENGTH')
    [print(f'{_pad(LONGEST_WORDS[i])}'.ljust(3*__) + f'{LENS[i]}') for i in range(n)]


    del _pad, LONGEST_WORDS










