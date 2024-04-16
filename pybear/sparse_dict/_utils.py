# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import sys, inspect
import numpy as np
import sparse_dict as sd
from pybear.utils import get_module_name

"""
clean                 Remove any 0 values, enforce contiguous zero-indexed outer keys, build any missing final inner key placeholders.
get_shape???
inner_len_quick       Length of inner dictionaries that are held by the outer dictionary assuming clean sparse dict.
inner_len             Length of inner dictionaries that are held by the outer dictionary.
outer_len             Length of the outer dictionary that holds the inner dictionaries as values, aka number of inner dictionaries
size_                 Return outer dict length * inner dict length.
shape_                Return (outer dict length, inner dict length) as tuple.
"""



def get_shape(name, OBJECT, given_orientation):
    # ALWAYS RETURNS (rows, columns) REGARDLESS OF ORIENTATION -
    # CONTRAST W NP THAT ALWAYS RETURNS (NUMBER OF INNER, LEN OF INNER)

    """Return shape in numpy format for list-types and sparse dicts, except
    (x,y) x is always rows and y is always columns w-r-t to given orientation."""

    _module = get_module_name(str(sys.modules[__name__]))
    _fxn = inspect.stack()[0][3]

    def _exception(words):
        raise Exception(f'{_module}.{_fxn}() >>> {words}')

    if name is None or not isinstance(name, str):
        raise TypeError(f'name MUST BE PASSED AND MUST BE A str')

    if OBJECT is None:
        return ()

    # if not OBJECT is None
    is_list = isinstance(OBJECT, (np.ndarray, list, tuple))
    is_dict = isinstance(OBJECT, dict)

    # IF OBJECT IS PASSED AS SINGLE [] OR {}, DONT WORRY ABOUT ORIENTATION,
    # ( CANT RETURN (,x) ONLY (x,) )
    if is_list: _shape = np.array(OBJECT).shape
    elif is_dict: _shape = sd.shape_(OBJECT)
    else: raise Exception(f'PASSED {name} OBJECT IS INVALID TYPE {type(OBJECT)}.')
    if len(_shape)==1: return _shape

    # CAN ONLY GET HERE IF len(_shape) >= 2
    if given_orientation is None:
        _exception(f'given_orientation MUST BE PASSED, CANNOT BE None.')
    if not given_orientation in ['ROW', 'COLUMN']:
        _exception(f'INVALID {name} given_orientation "{given_orientation}", '
                   f'MUST BE "ROW" OR "COLUMN".')

    #if given_orientation == 'ROW': _rows, _columns = _shape    _shape STAYS AS _shape
    if given_orientation == 'COLUMN': _shape = _shape[1], _shape[0]       # _shape GETS REVERSED

    del _exception; return _shape


def outer_len(DICT1):
    """Length of the outer dictionary that holds the inner dictionaries as
    values, aka number of inner dictionaries."""
    # DONT BOTHER TO clean() OR resize() HERE, SINCE ONLY THE SCALAR LENGTH IS
    # RETURNED (CHANGES TO DICTX ARENT RETAINED)
    fxn = inspect.stack()[0][3]
    DICT1 = dict_init(DICT1, fxn)
    insufficient_dict_args_1(DICT1, fxn)
    if DICT1 == {}: return (0,)
    try: outer_len = len(DICT1)
    except: raise Exception(f'Sparse dictionary is a zero-len outer dictionary in {module_name()}.{fxn}()')

    return outer_len


def inner_len_quick(DICT1):
    '''Length of inner dictionaries that are held by the outer dictionary assuming clean sparse dict.'''
    return max(DICT1[0]) + 1


def inner_len(DICT1):
    '''Length of inner dictionaries that are held by the outer dictionary.'''
    # DONT BOTHER TO clean() OR resize() HERE, SINCE ONLY THE SCALAR LENGTH IS RETURNED (CHANGES TO DICTX ARENT RETAINED)
    fxn = inspect.stack()[0][3]
    DICT1 = dict_init(DICT1, fxn)
    insufficient_dict_args_1(DICT1, fxn)
    if False not in np.fromiter(
            (True if DICT1[outer_idx] == {} else False for outer_idx in DICT1.keys()), dtype=bool):
        return 0
    # try: _inner_len = np.max(np.fromiter((max(DICT1[_]) + 1 for _ in DICT1), dtype=int))
    try: _inner_len = max(map(max, DICT1.values())) + 1
    except: raise Exception(f'Sparse dictionary has a zero-len inner dictionary in {module_name()}.{fxn}()')

    return _inner_len





def size_(DICT1):
    '''Return outer dict length * inner dict length.'''
    # DONT NEED arg CHECKS HERE, HANDLED BY outer_len() & inner_len()
    return outer_len(DICT1) * inner_len(DICT1)


def shape_(DICT1):
    """Returns shape of non-ragged sparse dict using numpy shape rules."""
    if DICT1 is None: return ()
    if DICT1 == {}: return (0,)

    # ASSUMES PLACEHOLDER RULE IS USED (len = max key + 1)
    if is_sparse_inner(DICT1): return (max(DICT1.keys())+1,)
    # AFTER HERE ALL MUST BE DICT OF INNER DICTS
    # IF ALL INNERS ARE {}  (ZERO len) #############################
    _LENS = list(map(len, DICT1.values()))
    if (min(_LENS)==0) and (max(_LENS)==0): return len(DICT1), 0
    del _LENS
    # END IF ALL INNERS ARE {}  (ZERO len) ##########################
    # TEST FOR RAGGEDNESS ###########################################
    _INNER_LENS = list(map(max, DICT1.values()))
    if not min(_INNER_LENS)==max(_INNER_LENS): raise Exception(f'sparse_dict.shape_() >>> DICT1 IS RAGGED.')
    # END TEST FOR RAGGEDNESS ###########################################
    return (len(DICT1), _INNER_LENS[0]+1)


def clean(DICT1):
    '''Remove any 0 values, enforce contiguous zero-indexed outer keys, build any missing final inner key placeholders.'''

    if DICT1 == {}: return DICT1

    DICT1 = dict_init(DICT1)
    insufficient_dict_args_1(DICT1, inspect.stack()[0][3])

    # CHECK IF KEYS START AT ZERO AND ARE CONTIGUOUS
    # FIND ACTUAL len, SEE IF OUTER KEYS MATCH EVERY POSN IN THAT RANGE
    # IF NOT, REASSIGN keys TO DICT, IDXed TO 0
    if False in np.fromiter((_ in DICT1 for _ in range(len(DICT1))), dtype=bool):
        for _, __ in enumerate(list(DICT1.keys())):
            DICT1[int(_)] = DICT1.pop(__)

    max_inner_key = int(np.max(np.fromiter((max(DICT1[outer_key]) for outer_key in DICT1),dtype=int)))
    for outer_key in DICT1:
        # ENSURE INNER DICT PLACEHOLDER RULE (KEY FOR LAST POSN, EVEN IF VALUE IS ZERO) IS ENFORCED
        DICT1[int(outer_key)][int(max_inner_key)] = DICT1[outer_key].get(max_inner_key, 0)
        # ENSURE THERE ARE NO ZERO VALUES IN ANY INNER DICT EXCEPT LAST POSITION
        VALUES_AS_NP = np.fromiter(DICT1[outer_key].values(), dtype=float)
        if 0 in VALUES_AS_NP[:-1]:
            KEYS_OF_ZEROES = np.fromiter((DICT1[outer_key]), dtype=int)[np.argwhere(VALUES_AS_NP[:-1]==0).transpose()[0]]
            np.fromiter((DICT1[outer_key].pop(_) for _ in KEYS_OF_ZEROES), dtype=object)
            del KEYS_OF_ZEROES

    del VALUES_AS_NP

    return DICT1







