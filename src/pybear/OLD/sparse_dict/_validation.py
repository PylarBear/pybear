# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import numpy as np
import pandas as pd
from pybear.utilities._get_module_name import get_module_name




def _module_name(sys_modules_str:str):
    """Return module name."""

    return get_module_name(sys_modules_str)


def _is_int(value:[int, float]):
    """Verify integer."""

    try:
        float(value)
    except:
        raise TypeError(f'must be an integer')

    if 'INT' not in str(type(value)).upper():
        raise TypeError(f'must be an integer.')


def _gt_zero(value:[int, float]):
    """Verify number is greater than zero."""

    err_msg = f'must be a number greater than zero.'
    try:
        float(value)
    except:
        raise TypeError(err_msg)

    if not value > 0:
        raise ValueError(err_msg)



def _list_init(LIST1:list=None, LIST_HEADER1:list=None):
    """List-type handling."""

    if LIST_HEADER1 is None: LIST_HEADER1 = [[]]

    if LIST1 is None: LIST1 = list()
    else: _list_check(LIST1)

    return np.array(LIST1), LIST_HEADER1


def _dict_init(DICT1:dict=None):
    """Sparse dictionary handling."""

    if DICT1 is None:
        DICT1 = dict()
    elif _is_sparse_outer(DICT1):
        _sparse_dict_check(DICT1)
    elif _is_sparse_inner(DICT1):
        _sparse_dict_check({0: DICT1})
    elif isinstance(DICT1, dict):
        raise ValueError(f'Invalid sparse dictionary format')
    else:
        raise TypeError(f'Invalid dtype {type(DICT1)} passed')

    return DICT1


def _datadict_init(DATADICT1:dict=None):
    """Datadict handling."""

    if DATADICT1 is None:
        DATADICT1 = dict()
        DATADICT_HEADER1 = [[]]
    else:
        _datadict_check(DATADICT1)
        # BUILD DATADICT_HEADER1 AND REKEY DATADICT OUTER KEYS NOW TO LOCK IN
        # SEQUENCE AND NOT LET NON-INT KEY GET PAST HERE!
        DATADICT_HEADER1 = [[]]
        key_counter = 0
        for key in list(DATADICT1.keys()):
            DATADICT_HEADER1[0].append(key)
            DATADICT1[key_counter] = DATADICT1.pop(key)
            key_counter += 1

    return DATADICT1, DATADICT_HEADER1


def _dataframe_init(DATAFRAME1=None):
    """Dataframe handling."""

    if DATAFRAME1 is None:
        DATAFRAME1 = pd.DataFrame({})
        DATAFRAME_HEADER1 = [[]]
    else:
        _dataframe_check(DATAFRAME1)
        # BUILD DATAFRAME_HEADER1 AND REKEY DATAFRAME OUTER KEYS NOW TO LOCK IN
        # SEQUENCE AND NOT LET NON-INT KEY GET PAST HERE!
        DATAFRAME_HEADER1 = [[]]
        key_counter = 0
        for key in list(DATAFRAME1.keys()):
            DATAFRAME_HEADER1[0].append(key)
            DATAFRAME1[key_counter] = DATAFRAME1.pop(key)
            key_counter += 1

    return DATAFRAME1, DATAFRAME_HEADER1



def _list_check(LIST1:list):
    """Require that LIST1 arg is a 1 or 2 dimensional list-type and is not ragged.
    """

    err_msg = (f'LIST arg must be a non-ragged array-like of 1 or 2 dimensions '
               f'that can be converted to an ndarray')

    try:
        iter(LIST1)
    except:
        raise TypeError(err_msg)

    if isinstance(LIST1, (dict, str)):
        raise TypeError(err_msg)

    try:
        LIST1 = np.array(list(LIST1))
    except:
        raise ValueError(err_msg)

    if len(LIST1.shape) not in  [1, 2]:
        raise ValueError(err_msg)


def _sparse_dict_check(DICT1:dict):
    """Require that objects to be processed as sparse dictionaries follow
    sparse dictionary rules.

    Parameters
    --------
    DICT1:
        any - object to be verified as is / is not a sparse dictionary.

    Returns
    ------
    Nothing

    """

    if not isinstance(DICT1, dict):
        raise TypeError(f'must be a dictionary')

    if len(DICT1) == 0:
        return None  # to escape out and bypass code below

    # reject ragged and handle empty inner dicts
    _inner_lens = np.unique(map(len, DICT1.values()))
    if len(_inner_lens) != 1:
        raise ValueError(f"ragged sparse dictionary")
    elif len(_inner_lens) == 1 and _inner_lens[0] == 0:
        return None # to escape out and bypass code below

    del _inner_lens

    # OUTER KEYS ARE INTEGERS
    err_msg = f"all sparse dictionary keys must be positive integers"
    # OUTER KEYS ARE PYTHON INTEGERS
    if any(map(isinstance, DICT1, (bool for _ in DICT1))):
        raise TypeError(err_msg)

    if not all(map(isinstance, DICT1, (int for _ in DICT1))):
        raise TypeError(err_msg)

    # OUTER KEYS ARE POSITIVE
    if min(DICT1) < 0:
        raise ValueError(f"all sparse dictionary keys must be positive integers")

    # IF IS INNER DICT ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
    try:
        list(map(float, DICT1.values()))
        # WILL EXCEPT IF DICT1.values() ARE DICTS
        # SINCE ESTABLISHED OUTER KEYS ARE INTS AND CAN MAP float TO VALUES,
        # THIS IS A GOOD INNER DICT
        return None # to escape out and bypass code below
    except:
        pass

    # END IF IS INNER DICT ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

    # IF IS OUTER DICT ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    # INNER OBJECTS ARE DICTS
    if not all(map(isinstance, DICT1.values(), (dict for _ in DICT1))):
        raise TypeError(f'must be a dictionary with values that are '
                        f'dictionaries')

    err_msg = lambda x: (f'outer index {x}: all inner keys must be positive '
                         f'integers')

    for outer_idx in DICT1:

        # INNER KEYS ARE INTEGERS
        # INNER KEYS ARE INTEGERS

        if any(map(isinstance, DICT1[outer_idx], (bool for _ in DICT1[outer_idx]))):
            raise TypeError(err_msg(outer_idx))

        if not all(map(isinstance, DICT1[outer_idx], (int for _ in DICT1[outer_idx]))):
            raise TypeError(err_msg(outer_idx))

        # INNER KEYS ARE POSITIVE
        if len(DICT1[outer_idx]) > 0 and min(DICT1[outer_idx]) < 0:
            raise ValueError(err_msg(outer_idx))

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

    del err_msg
    try:
        del values
    except:
        pass

    # HAS PLACEHOLDERS
    try:
        INNER_KEYS = list(map(list, DICT1.values()))
        max_inner_key = np.hstack((INNER_KEYS)).max()
        err_msg = (f'all inner dicts must have an actual value or a place-holding '
                   f'zero that demarcates the length of the vector')
        if not all(map(lambda x: max_inner_key in x, INNER_KEYS)):
            raise UnicodeError(err_msg)

        if not all(map(lambda x: max(x) == max_inner_key, INNER_KEYS)):
            raise UnicodeError(err_msg)

        del INNER_KEYS, max_inner_key, err_msg
    except UnicodeError as e:
        raise ValueError(e) from None
    except ValueError:
        pass
    # END IF IS OUTER DICT ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **


def _is_sparse_outer(DICT1:dict):
    """Returns True if object is an outer sparse dictionary, False otherwise."""

    if not isinstance(DICT1, dict):
        return False

    if DICT1 == {}:
        return False

    if not all(map(isinstance, DICT1.values(), [dict for values in DICT1])):
        return False

    if min(list(DICT1)) < 0:
        return False

    return True


def _is_sparse_inner(DICT1) -> bool:
    """Returns True if object is an inner sparse dictionary, False otherwise."""

    if not isinstance(DICT1, dict):
        return False

    if DICT1 == {}:
        return True

    if not all(map(lambda x: 'INT' in str(type(x)).upper(), DICT1)):
        return False

    if min(list(DICT1)) < 0:
        return False

    try:
        list(map(float, DICT1.values()))
        return True
    except:
        return False


def _datadict_check(DATADICT1:dict):
    """Require that objects to be processed as data dictionaries follow data
    dict rules: dictionary with list-type as values."""

    if not isinstance(DATADICT1, dict):
        raise TypeError(f'dictionary required as input.')

    for _ in DATADICT1.values():
        err_msg = f'input must be a dictionary with values that are vector-like.'
        try:
            iter(_)
        except:
            raise TypeError(err_msg)

        if isinstance(_, (dict, str)):
            raise TypeError(err_msg)

        if not np.array_equiv(np.array(_).ravel(), _):
            raise ValueError(err_msg)


def _dataframe_check(DATAFRAME1):
    """Verify DATAFRAME arg is a dataframe."""

    if 'DATAFRAME' not in str(type(DATAFRAME1)).upper():
        raise TypeError(f'input must be a DataFrame.')


def _insufficient_list_args_1(LIST1:list):
    """Verify LIST arg is satisfied when processing a function that requires a
    list.
    """

    err_msg = f'one list-type arg is required'
    if isinstance(LIST1, (dict, str)):
        raise TypeError(err_msg)

    try:
        iter(LIST1)
    except:
        raise TypeError(err_msg)


def _insufficient_dict_args_1(DICT1:dict):
    """Verify DICT1 arg is satisfied when processing a function that requires
    one dictionary.
    """

    if not isinstance(DICT1, dict):
        raise TypeError(
            f'one dictionary arg is required'
        )


def _insufficient_dict_args_2(DICT1:dict, DICT2:dict):
    """Verify DICT1 and DICT2 args are satisfied when processing a function that
    requires two dictionaries.
    """

    if not isinstance(DICT1, dict) or not isinstance(DICT2, dict):
        raise TypeError(f'requires two dictionary args')


def _insufficient_datadict_args_1(DATADICT1:dict):
    """Verify DATADICT1 arg is satisified when processing a function that requires
    one data dictionary.
    """

    if not isinstance(DATADICT1, dict):
        raise TypeError(f'one dictionary arg is required')

    _datadict_check(DATADICT1)


def _insufficient_dataframe_args_1(DATAFRAME1):
    """Verify DATAFRAME arg is satisfied when processing a function that requires
    one dataframe.
    """

    if "DATAFRAME" not in str(type(DATAFRAME1)).upper():
        raise TypeError(f'one dataframe arg is required')

    try:
        _size = DATAFRAME1.size.compute()
    except:
        _size = DATAFRAME1.size

















