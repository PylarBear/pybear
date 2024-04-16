# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import numpy as np
import pandas as pd
import joblib
import warnings
from pybear.utils import get_module_name



"""
__init__ CHECKS & EXCEPTIONS ##########################################################################################################
module_name                     Return file name.
list_init                       Pre-run list config.
dict_init                       Pre-run dict config.
datadict_init                   Pre-run datadict config.
dataframe_init                  Pre-run dataframe config.
list_check                      Require that LIST arg is a list-type, and is not ragged.
_sparse_dict_check               Require that objects to be processed as sparse dictionaries follow sparse dictionary rules.
_is_sparse_outer                 Returns True if object is an outer sparse dictionary
_is_sparse_inner                 Returns True if object is an inner sparse dictionary
datadict_check                  Require that objects to be processed as data dictionaries follow data dict rules: dictionary with list-type as values.
dateframe_check                 Verify DATAFRAME arg is a dataframe.
END __init__ CHECKS & EXCEPTIONS ######################################################################################################
RUNTIME CHECKS & EXCEPTIONS ###########################################################################################################
non_int                         Verify integer.
insufficient_list_args_1        Verify LIST arg is filled when processing a function that requires a list.
insufficient_dict_args_1        Verify DICT1 arg is filled when processing a function that requires one dictionary.
insufficient_dict_args_2        Verify DICT1 and DICT2 args are filled when processing a function that requres two dictionaries.
insufficient_datadict_args_1    Verify DATADICT1 arg is filled when processing a function that requires one data dictionary.
insufficient_dataframe_args_1   Verify DATAFRAME arg is filled when processing a function that requires one dataframe.
dot_size_check                  Verify two vectors are sparse dicts that both have unitary outer length and equal inner length.
broadcast_check                 Verify two sparse dicts follow standard matrix multiplication rules (m, n) x (j, k) ---> n == j.
matrix_shape_check              Verify two sparse dicts have equal outer and inner length.
outer_len_check                 Verify two sparse dicts have equal outer length.
inner_len_check                 Verify two sparse dicts have equal inner length.
symmetric_matmul_check          Verify two sparse dicts will matrix multiply to a symmetric matrix.
END RUNTIME CHECKS & EXCEPTIONS #######################################################################################################
"""

def module_name(sys_modules_str):
    '''Return file name.'''
    return get_module_name(sys_modules_str)


def list_init(LIST1=None, LIST_HEADER1=None, fxn=''):

    if LIST_HEADER1 is None: LIST_HEADER1 = [[]]

    if LIST1 is None: LIST1 = list()
    else: list_check(LIST1, fxn)

    return np.array(LIST1), LIST_HEADER1


def dict_init(DICT1=None, fxn=''):
    if DICT1 is None: DICT1 = dict()
    else: _sparse_dict_check(DICT1)

    return DICT1


def datadict_init(DATADICT1=None, fxn=''):

    if DATADICT1 is None:
        DATADICT1 = dict()
        DATADICT_HEADER1 = [[]]
    else:
        datadict_check(DATADICT1, fxn)
        # BUILD DATADICT_HEADER1 AND REKEY DATADICT OUTER KEYS NOW TO LOCK IN SEQUENCE AND NOT LET NON-INT KEY GET PAST HERE!
        DATADICT_HEADER1 = [[]]
        key_counter = 0
        for key in list(DATADICT1.keys()):
            DATADICT_HEADER1[0].append(key)
            DATADICT1[key_counter] = DATADICT1.pop(key)
            key_counter += 1

    return DATADICT1, DATADICT_HEADER1


def dataframe_init(DATAFRAME1=None, fxn=''):

    if DATAFRAME1 is None:
        DATAFRAME1 = pd.DataFrame({})
        DATAFRAME_HEADER1 = [[]]
    else:
        dataframe_check(DATAFRAME1, fxn)
        # BUILD DATAFRAME_HEADER1 AND REKEY DATAFRAME OUTER KEYS NOW TO LOCK IN SEQUENCE AND NOT LET NON-INT KEY GET PAST HERE!
        DATAFRAME_HEADER1 = [[]]
        key_counter = 0
        for key in list(DATAFRAME1.keys()):
            DATAFRAME_HEADER1[0].append(key)
            DATAFRAME1[key_counter] = DATAFRAME1.pop(key)
            key_counter += 1

    return DATAFRAME1, DATAFRAME_HEADER1

#########################################################################################################################################
#########################################################################################################################################
# __init__ CHECKS & EXCEPTIONS ##########################################################################################################

def list_check(LIST1, fxn):
    '''Require that LIST arg is a list-type, and is not ragged.'''
    if not isinstance(LIST1, (list,tuple,np.ndarray)):
        raise Exception(f'{module_name()}.{fxn}() requires LIST arg be a list, array, or tuple of lists, arrays, or '
                            f'tuples, i.e. [[],[]] or ((),()). Cannot be set or dictionary.')

    LIST1 = np.array(LIST1, dtype=object)

    with np.errstate(all='ignore'):
        if LIST1.size == 0:
            raise Exception(f'{module_name()}.{fxn}() LIST1 arg is an empty list-type.')

    try:
        _LENS = list(map(len, LIST1))
        if min(_LENS) != max(_LENS):
            raise Exception(f'{module_name()}.{fxn}() requires non-ragged list-type of list-types.  '
                            f'Min={min(_LENS)}, Max = {max(_LENS)}.')
    except:
        pass


def _sparse_dict_check(DICT1):
    """Verify an object is constructed by sparse dictionary rules.

    Parameters
    --------
    DICT1: any - object to be verified as is / is not a sparse dictionary.


    Returns
    ------
    Nothing
    """


    # BEAR
    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore")

    # IS DICT
    if not isinstance(DICT1, dict):
        raise TypeError(f'must be a dictionary')

    if len(DICT1) == 0:
        raise ValueError(f'empty dictionary')

    # INNER OBJECTS ARE DICTS
    if not all(map(isinstance, DICT1.values(), (dict for _ in DICT1))):
        raise TypeError(f'must be a dictionary with '
                        f'values that are dictionaries')

    # OUTER KEYS ARE INTEGERS
    if not all(map(lambda x: 'INT' in str(type(x)).upper(), list(DICT1))):
        raise TypeError(f'all outer keys must be integers')


    for outer_idx in DICT1:

        # INNER KEYS ARE INTEGERS
        if not all(map(lambda x: 'INT' in str(type(x)).upper(),
                       list(DICT1[outer_idx]))):
            raise TypeError(f'outer index {outer_idx}: '
                            f'all inner keys must be integers')

        # VALUES ARE NUMBERS INCL BOOL & np.nan, BUT NOT None
        values = DICT1[outer_idx].values()
        if None in values:
            raise TypeError(
                f'outer index {outer_idx}: sparse dict values cannot be '
                f'None'
            )

        try:
            np.fromiter(values, dtype=np.float64)
        except:
            raise TypeError(
                f'outer index {outer_idx}: all sparse dict values must be '
                f'numbers, booleans, or np.nan'
            ) from None

    del values

    # HAS PLACEHOLDERS
    max_inner_key = np.hstack((list(map(list, DICT1.values())))).max()
    if not all(map(lambda x: max_inner_key in x, map(list, DICT1.values()))):
        raise ValueError(f'all inner dicts must have an actual value or '
            f'a place-holding zero that demarcates the length of the vector')

    del max_inner_key


def _is_sparse_outer(DICT1):
    '''Returns True if object is an outer sparse dictionary.'''

    if not isinstance(DICT1, dict): return False
    if DICT1 == {}: return True
    first_key = np.fromiter(DICT1.keys(), dtype=np.int32)[0]
    if isinstance(DICT1[first_key], dict): return True
    else: return False


def _is_sparse_inner(DICT1):
    '''Returns True if object is an inner sparse dictionary.'''
    if not isinstance(DICT1, dict): return False
    if DICT1 == {}: return False
    first_key = np.fromiter(DICT1.keys(), dtype=np.int32)[0]
    try: int(first_key) == first_key
    except: return False
    if True in map(lambda x: x in str(type(DICT1[first_key])).upper(), ('INT', 'FLOAT')): return True
    else: return False




def datadict_check(DATADICT1, fxn):
    '''Require that objects to be processed as data dictionaries follow data dict rules: dictionary with list-type as values.'''
    if not isinstance(DATADICT1, dict):
        raise Exception(f'{module_name()}.{fxn}() requires dictionary as input.)')

    if DATADICT1 == {}:
        raise Exception(f'{module_name()}.{fxn}() input is an empty dictionary.)')

    for _ in DATADICT1.values():
        if not isinstance(_, (np.ndarray, list, tuple)):
            raise Exception(f'{module_name()}.{fxn}() requires input to be a dictionary with values that are list-types.')


def dataframe_check(DATAFRAME1, fxn):
    '''Verify DATAFRAME arg is a dataframe.'''
    if 'DATAFRAME' not in str(type(DATAFRAME1)).upper():
        raise Exception(f'{module_name()}.{fxn}() requires input to be a Pandas DataFrame.')

# END __init__ CHECKS & EXCEPTIONS ######################################################################################################
#########################################################################################################################################
#########################################################################################################################################

###############################################################################
###############################################################################
# RUNTIME CHECKS & EXCEPTIONS #################################################

def _is_int(value):
    # BEAR pytest IS WRITTEN
    '''Verify integer.'''

    try:
        float(value)
    except:
        raise TypeError(f'must be an integer')

    if 'INT' not in str(type(value)).upper():
        raise TypeError(f'must be an integer.')



def insufficient_list_args_1(LIST1, fxn):
    '''Verify LIST arg is filled when processing a function that requres a list.'''
    with np.errstate(all='ignore'):
        if not isinstance(LIST1, (np.ndarray, list, tuple)):
            raise Exception(f'{module_name()}.{fxn}() requires one list-type arg, LIST1.')
        if len(np.array(LIST1).reshape((1,-1))[0])==0:
            raise Exception(f'{module_name()}.{fxn}() input is an empty list-type.')


def insufficient_dict_args_1(DICT1, fxn):
    '''Verify DICT1 arg is filled when processing a function that requres one dictionary.'''
    if not isinstance(DICT1, dict):
        raise Exception(f'{module_name()}.{fxn}() requires one dictionary arg, DICT1.')
    if DICT1 == {}:
        raise Exception(f'{module_name()}.{fxn}() input is an empty dictionary.')


def insufficient_dict_args_2(DICT1, DICT2, fxn):
    '''Verify DICT1 and DICT2 args are filled when processing a function that requres two dictionaries.'''
    if not isinstance(DICT1, dict) or not isinstance(DICT2, dict):
        raise Exception(f'{module_name()}.{fxn}() requires two dictionary args, DICT1 and DICT2.')
    if DICT1 == {} or DICT2 == {}:
        raise Exception(f'{module_name()}.{fxn}() has at least one empty dictionary arg, DICT1 or DICT2.')


def insufficient_datadict_args_1(DATADICT1, fxn):
    '''Verify DATADICT1 arg is filled when processing a function that requres one data dictionary.'''
    if not isinstance(DATADICT1, dict):
        raise Exception(f'{module_name()}.{fxn}() requires one dictionary arg, DATADICT1.')
    if DATADICT1 == {}:
        raise Exception(f'{module_name()}.{fxn}() input is an empty dictionary.')


def insufficient_dataframe_args_1(DATAFRAME1, fxn):
    '''Verify DATAFRAME arg is filled when processing a function that requres one dataframe.'''
    if "DataFrame" not in str(type(DATAFRAME1)):
        raise Exception(f'{module_name()}.{fxn}() requires one DataFrame arg, DATAFRAME1.')
    if DATAFRAME1.equals(p.DataFrame({})):
        raise Exception(f'{module_name()}.{fxn}() input is an empty dictionary.')


def dot_size_check(DICT1, DICT2, fxn):
    '''Verify two vectors are sparse dicts that both have unitary outer length and equal inner length.'''

    if len(DICT1) != 1 or len(DICT2) != 1:
        raise Exception(f'{module_name()}.{fxn}() requires dictionaries with one integer key, one dict as values.)')
    if inner_len(DICT1) != inner_len(DICT2):
        raise Exception(f'{module_name()}.{fxn}() requires 2 dictionaries of equal stated length (last keys are equal).)')


def broadcast_check(DICT1, DICT2, fxn):        # DO THIS BEFORE TRANSPOSING DICT 2
    '''Verify two sparse dicts follow standard matrix multiplication rules (m, n) x (j, k) ---> n == j.'''
    if inner_len(DICT1) != outer_len(DICT2):
        raise Exception(f'{module_name()}.{fxn}() requires for DICT1(m x n) and DICT2(j x k) that num inner keys (n) of\n'
                        f'DICT1 == num outer keys (j) of DICT2 ---- (m, n) x (j, k) --> (m, k)\n'
                        f'{inner_len(DICT1)} is different than {outer_len(DICT2)}.')


def matrix_shape_check(DICT1, DICT2, fxn):
    '''Verify two sparse dicts have equal outer and inner length.'''
    _shape1, _shape2 = shape_(DICT1), shape_(DICT2)
    if _shape1 != _shape2:
        raise Exception(f'{module_name()}.{fxn}() requires both sparse dicts to be equally sized.  Dict 1 is {_shape1[0]} x '
                        f'{_shape1[1]} and Dict 2 is {_shape2[0]} x {_shape2[1]}')


def outer_len_check(DICT1, DICT2, fxn):
    '''Verify two sparse dicts have equal outer length.'''
    outer_len1 = outer_len(DICT1)
    outer_len2 = outer_len(DICT2)
    if outer_len1 != outer_len2:
        raise Exception(
            f'{module_name()}.{fxn}() requires both sparse dicts to have equal outer length.  Dict 1 is {outer_len1} '
            f'and Dict 2 is {outer_len2}')


def inner_len_check(DICT1, DICT2, fxn):
    '''Verify two sparse dicts have equal inner length.'''
    inner_len1 = inner_len(DICT1)
    inner_len2 = inner_len(DICT2)
    if inner_len1 != inner_len2:
        raise Exception(
            f'{module_name()}.{fxn}() requires both sparse dicts to have equal inner length.  Dict 1 is {inner_len1} '
            f'and Dict 2 is {inner_len2}')


def symmetric_matmul_check(DICT1, DICT2, DICT1_TRANSPOSE=None, DICT2_TRANSPOSE=None):
    '''Verify two sparse dicts will matrix multiply to a symmetric matrix.'''

    # GET DICT2_T FIRST JUST TO TEST DICT2 IS TRANSPOSE OF DICT1
    if not DICT2_TRANSPOSE is None: DICT2_T = DICT2_TRANSPOSE
    else: DICT2_T = sparse_transpose(DICT2)

    # TEST DICT2 IS TRANSPOSE OF DICT1
    DICT2 = sparse_transpose(DICT2)
    if safe_sparse_equiv(DICT1, DICT2_T): return True

    # IF DICT2 IS NOT TRANSPOSE OF DICT1, MAKE DICT1_T & PROCEED TO SYMMETRY TESTS
    if not DICT1_TRANSPOSE is None: DICT1_T = DICT1_TRANSPOSE
    else: DICT1_T = sparse_transpose(DICT1)

    # TEST BOTH DICT1 AND DICT2 ARE SYMMETRIC
    test2 = safe_sparse_equiv(DICT1, DICT1_T)
    if not test2: return False

    test3 = safe_sparse_equiv(DICT2, DICT2_T)
    if not test3: return False

    return True  # IF GET TO THIS POINT, MUST BE TRUE

# END RUNTIME CHECKS & EXCEPTIONS #######################################################################################################
#########################################################################################################################################
#########################################################################################################################################


