# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from copy import deepcopy
import numpy as np
from typing import Iterable, Union
from utilities._benchmarking import time_memory_benchmark as tmb
from feature_extraction.text import alphanumeric_str as ans


# TEST FOR TextCleaner.remove_characters()

"""
# UNKNOWN DATE, PERHAPS DONE ON WINDOWS
np.uniques GOTTEN FROM np.fromiter OVER CHARS IS HANDS DOWN WINNER
np_unique1     average, sdev: time = 0.695 sec, 0.078; mem = 0.000, 0.000
np_unique2     average, sdev: time = 2.586 sec, 0.140; mem = 0.000, 0.000
plug_n_chug    average, sdev: time = 2.443 sec, 0.206; mem = 0.000, 0.000

# 24_06_21 ON LINUX
np_unique1               time = 17.532 +/- 1.575 sec; mem = 0.000 +/- 0.000 MB
np_unique2               time = 13.533 +/- 0.125 sec; mem = 0.000 +/- 0.000 MB
np_unique3               time = 14.687 +/- 0.170 sec; mem = -10.400 +/- 2.498 MB
map_set                  time = 5.319 +/- 0.111 sec; mem = 0.200 +/- 0.400 MB
map_set_parallel_process time = 8.098 +/- 0.036 sec; mem = 23.000 +/- 2.966 MB
map_set_parallel_threads time = 21.916 +/- 0.701 sec; mem = 9.600 +/- 1.855 MB
plug_n_chug              time = 5.516 +/- 0.009 sec; mem = 0.000 +/- 0.000 MB

"""


# THIS MODULE TESTS SPEED OF DIFFERENT METHODS IN CLEANING CHARS OUT OF
# A LIST OF STRINGS [str1, str2,... ]
# METHODS 1, 2, 3: GET UNIQUES FROM EACH STR AND DO np.char.replace OVER
#   UNIQUES TO REMOVE DISALLOWED CHARS FROM EACH STR
# METHOD 4: USE set TO GET UNIQUES
# METHOD 5: PLUG-N-CHUG VIA for LOOP OVER EACH STR AND BUILD A SUBSTITUTE
# STR W ONLY ALLOWED CHARS

def np_unique1(
        LIST_OF_STRS: Iterable[str],
        *,
        allowed_chars:str=ans.alphanumeric_str(),
        disallowed_chars:str=None
    ) -> Iterable[str]:

    err_msg = f"can only pass one of 'allowed_chars' or 'disallowed_chars'"
    if allowed_chars is not None and disallowed_chars is not None:
        raise ValueError(err_msg)

    del err_msg

    _have_allowed = isinstance(allowed_chars, str)
    _have_disallowed = isinstance(disallowed_chars, str)

    for row_idx in range(len(LIST_OF_STRS)):
        UNIQUES = "".join(np.unique(list(LIST_OF_STRS[row_idx])))
        for char in UNIQUES:
            if (_have_allowed and char not in allowed_chars):
                LIST_OF_STRS[row_idx] = \
                    np.char.replace(LIST_OF_STRS[row_idx], char, '')
            elif (_have_disallowed and char in disallowed_chars):
                LIST_OF_STRS[row_idx] = \
                    np.char.replace(LIST_OF_STRS[row_idx], char, '')
    del UNIQUES
    
    return LIST_OF_STRS


def np_unique2(
        LIST_OF_STRS: Iterable[str],
        *,
        allowed_chars:str=ans.alphanumeric_str(),
        disallowed_chars:str=None
    ) -> Iterable[str]:


    err_msg = f"can only pass one of 'allowed_chars' or 'disallowed_chars'"
    if allowed_chars is not None and disallowed_chars is not None:
        raise ValueError(err_msg)

    del err_msg

    _have_allowed = isinstance(allowed_chars, str)
    _have_disallowed = isinstance(disallowed_chars, str)

    for row_idx in range(len(LIST_OF_STRS)):
        UNIQUES = "".join(np.unique(np.fromiter(str(LIST_OF_STRS[row_idx]), '<U1')))
        for char in UNIQUES:
            if (_have_allowed and char not in allowed_chars):
                LIST_OF_STRS[row_idx] = \
                    np.char.replace(LIST_OF_STRS[row_idx], char, '')
            elif (_have_disallowed and char in disallowed_chars):
                LIST_OF_STRS[row_idx] = \
                    np.char.replace(LIST_OF_STRS[row_idx], char, '')

    del UNIQUES

    return LIST_OF_STRS



def np_unique3(
        LIST_OF_STRS: Iterable[str],
        *,
        allowed_chars:str=ans.alphanumeric_str(),
        disallowed_chars:str=None
    ) -> Iterable[str]:

    err_msg = f"can only pass one of 'allowed_chars' or 'disallowed_chars'"
    if allowed_chars is not None and disallowed_chars is not None:
        raise ValueError(err_msg)

    del err_msg

    _have_allowed = isinstance(allowed_chars, str)
    _have_disallowed = isinstance(disallowed_chars, str)

    ALL_UNIQUES = "".join(
        np.unique(
            np.hstack((
                list(map(np.unique, list(map(list, LIST_OF_STRS))))
            ))
        )
    )

    for unq in ALL_UNIQUES:
        if _have_allowed and unq not in allowed_chars:
            LIST_OF_STRS = list(
                map(lambda x: np.char.replace(x, unq, ''), LIST_OF_STRS)
            )
        elif _have_disallowed and unq in disallowed_chars:
            LIST_OF_STRS = list(
                map(lambda x: np.char.replace(x, unq, ''), LIST_OF_STRS)
            )

    del ALL_UNIQUES

    return LIST_OF_STRS


def map_set(
        LIST_OF_STRS: Iterable[str],
        *,
        allowed_chars:str=ans.alphanumeric_str(),
        disallowed_chars:str=None
    ) -> Iterable[str]:

    err_msg = f"can only pass one of 'allowed_chars' or 'disallowed_chars'"
    if allowed_chars is not None and disallowed_chars is not None:
        raise ValueError(err_msg)

    del err_msg

    _have_allowed = isinstance(allowed_chars, str)
    _have_disallowed = isinstance(disallowed_chars, str)

    ALL_UNIQUES = \
        "".join(
            set(
                np.hstack((list(map(list, map(set, LIST_OF_STRS)))))
            )
    )

    for unq in ALL_UNIQUES:
        if _have_allowed and unq not in allowed_chars:
            LIST_OF_STRS = list(
                map(lambda x: np.char.replace(x, unq, ''), LIST_OF_STRS)
            )
        elif _have_disallowed and unq in disallowed_chars:
            LIST_OF_STRS = list(
                map(lambda x: np.char.replace(x, unq, ''), LIST_OF_STRS)
            )
    del ALL_UNIQUES

    return LIST_OF_STRS



def map_set_parallel_process(
        LIST_OF_STRS: Iterable[str],
        *,
        allowed_chars:str=ans.alphanumeric_str(),
        disallowed_chars:str=None,
        n_jobs:int=1
    ) -> Iterable[str]:

    import joblib

    err_msg = f"can only pass one of 'allowed_chars' or 'disallowed_chars'"
    if allowed_chars is not None and disallowed_chars is not None:
        raise ValueError(err_msg)

    del err_msg

    def parallel_remove_disallowed(
        _string: str,
        _allowed: Union[str, None],
        _disallowed: Union[str, None]
        ) -> str:

        INDIV_UNIQUES = set(_string)

        for unq in INDIV_UNIQUES:
            if isinstance(_allowed, str) and unq not in _allowed:
                _string = np.char.replace(_string, unq, '')
            elif isinstance(_disallowed, str) and unq in _disallowed:
                _string = np.char.replace(_string, unq, '')

        return _string

    # DONT HARD-CODE backend, ALLOW A CONTEXT MANAGER TO SET
    joblib_kwargs = {'prefer': 'processes', 'return_as': 'list', 'n_jobs': n_jobs}
    LIST_OF_STRS = joblib.Parallel(**joblib_kwargs)(
        joblib.delayed(parallel_remove_disallowed)(
            _string,
            allowed_chars,
            disallowed_chars
            ) for _string in LIST_OF_STRS
    )

    return LIST_OF_STRS


def map_set_parallel_threads(
        LIST_OF_STRS: Iterable[str],
        *,
        allowed_chars:str=ans.alphanumeric_str(),
        disallowed_chars:str=None,
        n_jobs:int=1
    ) -> Iterable[str]:

    import joblib

    err_msg = f"can only pass one of 'allowed_chars' or 'disallowed_chars'"
    if allowed_chars is not None and disallowed_chars is not None:
        raise ValueError(err_msg)

    del err_msg

    def parallel_remove_disallowed(
        _string: str,
        _allowed: Union[str, None],
        _disallowed: Union[str, None]
        ) -> str:

        INDIV_UNIQUES = set(_string)

        for unq in INDIV_UNIQUES:
            if isinstance(_allowed, str) and unq not in _allowed:
                _string = np.char.replace(_string, unq, '')
            elif isinstance(_disallowed, str) and unq in _disallowed:
                _string = np.char.replace(_string, unq, '')

        return _string

    # DONT HARD-CODE backend, ALLOW A CONTEXT MANAGER TO SET
    joblib_kwargs = {'prefer': 'threads', 'return_as': 'list', 'n_jobs': n_jobs}
    LIST_OF_STRS = joblib.Parallel(**joblib_kwargs)(
        joblib.delayed(parallel_remove_disallowed)(
            _string,
            allowed_chars,
            disallowed_chars
            ) for _string in LIST_OF_STRS
    )

    return LIST_OF_STRS



def plug_n_chug(
        LIST_OF_STRS: Iterable[str],
        *,
        allowed_chars:str=ans.alphanumeric_str(),
        disallowed_chars:str=None
    ) -> Iterable[str]:

    err_msg = f"can only pass one of 'allowed_chars' or 'disallowed_chars'"
    if allowed_chars is not None and disallowed_chars is not None:
        raise ValueError(err_msg)

    del err_msg

    _have_allowed = isinstance(allowed_chars, str)
    _have_disallowed = isinstance(disallowed_chars, str)

    for row_idx, _string in enumerate(LIST_OF_STRS):
        holder_str = f''
        for char_idx, char in enumerate(str(_string)):
            if _have_allowed and char in allowed_chars:
                holder_str += char
            elif _have_disallowed and char not in disallowed_chars:
                holder_str += char
        LIST_OF_STRS[row_idx] = holder_str

    del holder_str, char

    return LIST_OF_STRS









# TEST ACCURACY OF FUNCTIONS ** * ** * ** * ** * ** * ** * ** * ** * ** *

DUM_TXT = [
    (f"python - List of all unique characters in a string? - Stack "
        "Overflowhttps://stackoverflow.com › questions"),
    (f"› list-of-all-uniqu... Apr 25, 2017 — Now of course I have two solutions "
     f"in my mind. One is using a list"),
    (f"that will map the characters with their ASCII codes. So whenever I "
     f"encounter a letter it will ... 9 answers."),
    (f"Top answer: The simplest solution is probably: In [10]: "
     f"''.join(set('aaabcabccd')) Out[10]: 'a']")
]


TEST1 = np_unique1(deepcopy(DUM_TXT))
TEST2 = np_unique2(deepcopy(DUM_TXT))
TEST3 = np_unique3(deepcopy(DUM_TXT))
TEST4 = plug_n_chug(deepcopy(DUM_TXT))
TEST5 = map_set(deepcopy(DUM_TXT))
TEST6 = map_set_parallel_process(deepcopy(DUM_TXT), n_jobs=-1)
TEST7 = map_set_parallel_threads(deepcopy(DUM_TXT), n_jobs=-1)

if np.array_equiv(TEST1, TEST2) and np.array_equiv(TEST1, TEST3) and \
        np.array_equiv(TEST1, TEST4) and np.array_equiv(TEST1, TEST5) and \
        np.array_equiv(TEST1, TEST6) and np.array_equiv(TEST1, TEST7):
    print('\033[92mACCURACY TESTS PASSED.\033[0m')
else:
    _print = lambda x: print(f'\033[91m{x}\033[0m')
    _print(f'ACCURACY TEST FAILED')
    _print(f'np_unique1')
    [print(_) for _ in TEST1]
    print()
    _print(f'np_unique2')
    [print(_) for _ in TEST2]
    print()
    _print(f'np_unique3')
    [print(_) for _ in TEST3]
    print()
    _print(f'plug_n_chug')
    [print(_) for _ in TEST4]
    print()
    _print(f'map_set')
    [print(_) for _ in TEST5]
    print()
    _print(f'map_set_parallel_process')
    [print(_) for _ in TEST6]
    print()
    _print(f'map_set_parallel_threads')
    [print(_) for _ in TEST7]

del TEST1, TEST2, TEST3, TEST4, TEST5, TEST6, TEST7, DUM_TXT
# END TEST ACCURACY OF FUNCTIONS ** * ** * ** * ** * ** * ** * ** * ** *







DUM_TXT = [
    "Contrary to popular belief, Lorem Ipsum is not simply _random_ text.",
    ("It has roots in a piece of classical Latin literature from 45 BC, making "
     "it over 2000 years old."),
    ("Richard McClintock, a Latin professor at Hampden-Sydney College in "
     "Virginia, looked up one of the more obscure Latin words, consectetur, "
     "from a Lorem Ipsum passage, and going through the cites of the word "
     "in classical literature, discovered the undoubtable source."),
    ("Lorem Ipsum comes from sections 1.10.32 and 1.10.33 of "
     "'de Finibus Bonorum et Malorum' (The Extremes of Good and Evil) by Cicero "
     "written in 45 BC. This book is a treatise on the theory of ethics, very "
     "popular during the Renaissance."),
    ("The first line of Lorem Ipsum, 'Lorem ipsum dolor sit amet...' comes from "
     "a line in section 1.10.32.")
]


print(f'\n Building test data...')
# THIS GROWS EXPONENTIALLY!
# (5, 10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10240, 20480,
# 40960, 81920, 164000, 328000...)
for _ in range(15):
    DUM_TXT = np.hstack((DUM_TXT, DUM_TXT), dtype='<U1000')

del _

print(f'\n Running time / memory benchmarks...')

tmb(
    ('np_unique1', np_unique1, [DUM_TXT], {}),
    ('np_unique2', np_unique2, [DUM_TXT], {}),
    ('np_unique3', np_unique3, [DUM_TXT], {}),
    ('map_set', map_set, [DUM_TXT], {}),
    ('map_set_parallel_process', map_set_parallel_process, [DUM_TXT], {'n_jobs': -1}),
    ('map_set_parallel_threads', map_set_parallel_threads, [DUM_TXT], {'n_jobs': -1}),
    ('plug_n_chug', plug_n_chug, [DUM_TXT], {}),
    number_of_trials = 7,
    rest_time = 1,
    verbose = True
)





















