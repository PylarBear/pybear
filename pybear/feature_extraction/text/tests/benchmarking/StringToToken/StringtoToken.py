# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import time
import numpy as np
import joblib
from pybear.utils import time_memory_benchmark as tmb

# THE BIG TAKEAWAYS
# -- WITH TOKENS, NP dtype=object MEM IS COMPARABLE TO py, '<U' IS MUCH MORE
# -- FORGET ABOUT PARALLELISM
# -- list(map(str.split, X)) AND for LOOP ARE COMPARABLE, BETTER THAN NP
#       AND/OR JOBLIB IN ALL CASES



# TIME/MEM TRIALS 24_06_24 ON LINUX
#
# SIZE = 2_000 x 2_000, TRIALS = 7, REST = 1 (so small to get char_split to work)
# ALL OF THESE EXCEPT map_list ARE WITH np dtype=object, INSTEAD OF <U40 OR <U15.
# WHEN USING <U, MEMORIES ARE UP TO DOUBLE THESE, USE np dtype=object. FORGET
# ABOUT char_split.
# map_fromiter -> np/np      time = 0.464 +/- 0.102 sec; mem = 34.600 +/- 5.571 MB
# map_list -> py/py          time = 0.502 +/- 0.176 sec; mem = 35.000 +/- 6.033 MB
# for_loop -> np/np          time = 0.667 +/- 0.264 sec; mem = 35.000 +/- 6.033 MB
# n_jobs_processes -> np/np  time = 1.555 +/- 0.433 sec; mem = 38.000 +/- 6.099 MB
# n_jobs_threads -> np/np    time = 1.163 +/- 0.099 sec; mem = 41.600 +/- 4.630 MB
# char_split -> np/np        time = 0.940 +/- 0.032 sec; mem = 315.400 +/- 3.200 MB
#
## SIZE = 7_500 x 7_500, TRIALS = 7, REST = 1
# ALL OF THESE EXCEPT map_list ARE WITH np dtype=object
# map_fromiter -> np/np      time = 7.355 +/- 0.905 sec; mem = 3,049.200 +/- 34.672 MB
# map_list -> py/py          time = 6.596 +/- 0.585 sec; mem = 3,071.800 +/- 48.897 MB
# for_loop -> np/np          time = 7.307 +/- 0.043 sec; mem = 3,071.400 +/- 46.569 MB
# n_jobs_processes -> np/np  time = 16.304 +/- 0.318 sec; mem = 2,973.600 +/- 31.014 MB
# n_jobs_threads -> np/np    time = 15.142 +/- 0.577 sec; mem = 2,992.000 +/- 37.304 MB

## SIZE = 7_500 x 7_500, TRIALS = 15, REST = 1
# ALL OF THESE ARE py list / py list
# map_list -> py/py             time = 6.055 +/- 0.231 sec; mem = 3,130.818 +/- 41.615 MB
# for_loop -> py/py             time = 5.844 +/- 0.166 sec; mem = 3,118.545 +/- 41.712 MB
# n_jobs_processes -> py/py     time = 15.644 +/- 0.167 sec; mem = 2,952.364 +/- 39.287 MB
# n_jobs_threads -> py/py       time = 14.681 +/- 0.316 sec; mem = 3,373.364 +/- 24.031 MB

## SIZE = 400_000 ROWS x 100 COLS, TRIALS = 7, REST = 1
# ROWS >> COLUMNS
# ALL OF THESE ARE py list / py list
# map_list -> py/py             time = 7.197 +/- 0.288 sec; mem = 2,088.800 +/- 36.570 MB
# for_loop -> py/py             time = 7.529 +/- 0.095 sec; mem = 2,106.400 +/- 35.539 MB
# n_jobs_processes -> py/py     time = 19.439 +/- 0.220 sec; mem = 2,107.000 +/- 37.143 MB
# n_jobs_threads -> py/py       time = 62.250 +/- 1.158 sec; mem = 2,291.600 +/- 32.166 MB

## SIZE = 100 ROWS x 562_500 COLS, TRIALS = 7, REST = 1
# COLUMNS >> ROWS
# ALL OF THESE ARE py list / py list
# map_list -> py/py             time = 5.515 +/- 0.249 sec; mem = 3,555.000 +/- 52.949 MB
# for_loop -> py/py             time = 5.318 +/- 0.256 sec; mem = 3,534.000 +/- 48.220 MB
# n_jobs_processes -> py/py     time = 16.417 +/- 0.790 sec; mem = 3,464.000 +/- 145.715 MB
# n_jobs_threads -> py/py       time = 6.944 +/- 0.201 sec; mem = 3,500.400 +/- 65.713 MB





def map_fromiter(_CLEANED_TEXT):

    _is_list_of_strs = all(
        map(isinstance, _CLEANED_TEXT, (str for _ in _CLEANED_TEXT))
    )

    if not _is_list_of_strs:
        pass
    elif _is_list_of_strs:  # MUST BE LIST OF strs
        # ASSUME THE TEXT STRING CAN BE SEPARATED ON ' '

        __ = np.fromiter(
            map(np.fromiter,
                map(str.split, _CLEANED_TEXT),
                (object for _ in _CLEANED_TEXT)   # <U40
            ),
            dtype=object
        )

        return __





def map_list(_CLEANED_TEXT):

    _is_list_of_strs = all(
        map(isinstance, _CLEANED_TEXT, (str for _ in _CLEANED_TEXT))
    )

    if not _is_list_of_strs:
        pass
    elif _is_list_of_strs:  # MUST BE LIST OF strs
        # ASSUME THE TEXT STRING CAN BE SEPARATED ON ' '
        # __ = np.fromiter(
        #     map(str.split, _CLEANED_TEXT),
        #     dtype=object
        # )

        __ = list(map(str.split, _CLEANED_TEXT))

        return __




def for_loop(_CLEANED_TEXT):

    _is_list_of_strs = all(
        map(isinstance, _CLEANED_TEXT, (str for _ in _CLEANED_TEXT))
    )

    if not _is_list_of_strs:
        pass
    elif _is_list_of_strs:  # MUST BE LIST OF strs
        # ASSUME THE TEXT STRING CAN BE SEPARATED ON ' '
        __ = []
        for idx in range(len(_CLEANED_TEXT)):

            __.append(_CLEANED_TEXT[idx].split(sep=' '))

        #     __.append(
        #         np.array(_CLEANED_TEXT[idx].split(sep=' '), dtype=object)  # <U40
        #     )
        #
        # __ = np.array(__, dtype=object)

        return __




def n_jobs_processes(_CLEANED_TEXT, n_jobs=-1):

    _is_list_of_strs = all(
        map(isinstance, _CLEANED_TEXT, (str for _ in _CLEANED_TEXT))
    )

    if not _is_list_of_strs:
        pass
    elif _is_list_of_strs:  # MUST BE LIST OF strs
        # ASSUME THE TEXT STRING CAN BE SEPARATED ON ' '

        def _splitter(_string):
            # return np.array(_string.split(sep=' '), dtype=object)  # <U40
            return _string.split(sep=' ')

        joblib_kwargs = {'prefer': 'processes', 'return_as':'list', 'n_jobs': n_jobs}
        __ = joblib.Parallel(**joblib_kwargs)(
            joblib.delayed(_splitter)(_string) for _string in _CLEANED_TEXT
        )

        # __ = np.array(__, dtype=object)

        return __


def n_jobs_threads(_CLEANED_TEXT, n_jobs=-1):

    _is_list_of_strs = all(
        map(isinstance, _CLEANED_TEXT, (str for _ in _CLEANED_TEXT))
    )

    if not _is_list_of_strs:
        pass
    elif _is_list_of_strs:  # MUST BE LIST OF strs
        # ASSUME THE TEXT STRING CAN BE SEPARATED ON ' '

        def _splitter(_string):
            return _string.split(sep=' ')
            # return np.array(_string.split(sep=' '), dtype=object)  #<U40

        joblib_kwargs = {'prefer': 'threads', 'return_as':'list', 'n_jobs': n_jobs}
        __ = joblib.Parallel(**joblib_kwargs)(
            joblib.delayed(_splitter)(_string) for _string in _CLEANED_TEXT
        )

        # __ = np.array(__, dtype=object)

        return __


# def char_split(_CLEANED_TEXT_SQUARE):
#
#     _is_list_of_strs = all(
#         map(isinstance, _CLEANED_TEXT_SQUARE, (str for _ in _CLEANED_TEXT_SQUARE))
#     )
#
#     if not _is_list_of_strs:
#         pass
#     elif _is_list_of_strs:  # MUST BE LIST OF strs
#         # ASSUME THE TEXT STRING CAN BE SEPARATED ON ' '
#
#         __ = np.char.split(_CLEANED_TEXT_SQUARE).astype(object)#.tolist()
#
#         return __




print(f'building test object...')
_rows = 100
_cols = 562_500

_CLEANED_TEXT = []
for __ in range(_rows):
    _CLEANED_TEXT.append(
        ('loremipsum '* np.random.randint(int(_cols*.95), _cols))[:-1]
    )
print(f'done.')

run_accuracy_tests = False

if run_accuracy_tests:
    # check accuracy
    print(f'performing accuracy test...')

    out_map_fromiter = map_fromiter(_CLEANED_TEXT)
    assert isinstance(out_map_fromiter, np.ndarray)

    out_map_list = map_list(_CLEANED_TEXT)
    assert isinstance(out_map_list, list) #np.ndarray)
    # giving error, has to do with the shape
    # assert np.array_equiv(out_map_fromiter, out_map_list)
    for idx in range(len(_CLEANED_TEXT)):
        assert np.array_equiv(out_map_fromiter[idx], out_map_list[idx])
    del out_map_list
    time.sleep(5)

    out_for_loop = for_loop(_CLEANED_TEXT)
    assert isinstance(out_for_loop, list) #np.ndarray)
    # giving error, has to do with the shape
    # assert np.array_equiv(out_map_fromiter, out_for_loop)
    for idx in range(len(_CLEANED_TEXT)):
        assert np.array_equiv(out_map_fromiter[idx], out_for_loop[idx])
    del out_for_loop
    time.sleep(5)

    out_n_jobs_processes = n_jobs_processes(_CLEANED_TEXT)
    assert isinstance(out_n_jobs_processes, list) #np.ndarray)
    # giving error, has to do with the shape
    # assert np.array_equiv(out_map_fromiter, out_n_jobs_processes)
    for idx in range(len(_CLEANED_TEXT)):
        assert np.array_equiv(out_map_fromiter[idx], out_n_jobs_processes[idx])
    del out_n_jobs_processes
    time.sleep(5)

    out_n_jobs_threads = n_jobs_threads(_CLEANED_TEXT)
    assert isinstance(out_n_jobs_threads, list) #np.ndarray)
    # giving error, has to do with the shape
    # assert np.array_equiv(out_map_fromiter, out_n_jobs_threads)
    for idx in range(len(_CLEANED_TEXT)):
        assert np.array_equiv(out_map_fromiter[idx], out_n_jobs_threads[idx])
    del out_n_jobs_threads

    # out_char_split = char_split(_CLEANED_TEXT)
    # assert isinstance(out_char_split, np.ndarray)
    # # giving error, has to do with the shape
    # # assert np.array_equiv(out_map_fromiter, out_n_jobs_threads)
    # for idx in range(len(_CLEANED_TEXT)):
    #     assert np.array_equiv(out_map_fromiter[idx], out_char_split[idx])
    # del out_char_split


    del out_map_fromiter
    time.sleep(5)
    print(f'done.')



tmb(
    # ('map_fromiter', map_fromiter, [_CLEANED_TEXT], {}),
    ('map_list', map_list, [_CLEANED_TEXT], {}),
    ('for_loop', for_loop, [_CLEANED_TEXT], {}),
    ('n_jobs_processes', n_jobs_processes, [_CLEANED_TEXT], {'n_jobs': -1}),
    ('n_jobs_threads', n_jobs_threads, [_CLEANED_TEXT], {'n_jobs': -1}),
    # ('char_split', char_split, [[('lorem ipsum '*_rows)[:-1] for __ in range(_rows)]], {}),
    number_of_trials=7,
    rest_time=1,
    verbose=True
)





