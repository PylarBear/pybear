import sys, inspect, warnings, time
import numpy as np, pandas as pd
from copy import deepcopy
from functools import wraps

from utilities._get_module_name import get_module_name
from pybear.data_validation import (
    validate_user_input as vui,
    arg_kwarg_validater as akv
)
from pybear.sparse_dict._utils import (
    outer_len
)
from pybear.sparse_dict import get_shape as gs
from pybear.sparse_dict._validation import (
    _dict_init,
    _insufficient_dict_args_1
)


# *******
# ******* 10/11/22 DO NOT PUT np dtypes AS DICT KEYS!!!! ONLY USE py int.  DO NOT PUT np.float64 AS VALUE dtype!!!!
# ******* ONLY USE py float!!!  NON py NATIVE dtypes CRIPPLE DICT PROCESSING SPEED.  MEASUREMENTS INDICATE BY AT LEAST 50%
# *******



# FOR 500 COLS x 1000 ROWS:
# WHEN NOT SYMMETRIC W/ 10% SPARSE, NP.MATMUL = 36.62s, DENSE DOTS = 50.34s, DICT DOTS = 13.04s
# WHEN NOT SYMMETRIC W/ 90% SPARSE, NP.MATMUL = 35.41s, DENSE DOTS = 48.32s, DICT DOTS = 4.04s
# WHEN SYMMETRIC W/ 10% SPARSE,     NP.MATMUL = 36.97s, DENSE DOTS = 29.17s, DICT DOTS = 7.76s
# WHEN SYMMETRIC W/ 90% SPARSE,     NP.MATMUL = 34.60s, DENSE DOTS = 25.02s, DICT DOTS = 2.51s

'''

CREATION, HANDLING & MAINTENANCE ######################################################################################################



resize_inner                    Resize sparse dict to user-entered inner dict length.  Reducing size may truncate non-zero values;
                                increasing size will introduce zeros (empties) and original inner size placeholder rule (entry for last item even if 0) holds.
#resize_outer                   Resize sparse dict to user-entered outer dict length.  Reducing size may truncate non-zero values;
                                increasing size will introduce zeros (placeholder inner dicts) and original outer size placeholder rule holds.
#resize                         Resize sparse dict to user-entered (len outer dict, len inner dicts) dimensions.  Reducing size may truncate non-zero values;
                                increasing size will introduce zeros (empties in inner dicts, placeholder in outer dict) and original size placeholder rules hold.
#merge_outer                    Merge outer dictionaries of 2 dictionaries with safeguards.  Inner dictionary lengths must be equal.
#core_merge_outer               Merge outer dictionaries of 2 dictionaries without safeguards.  Inner dictionary lengths must be equal.
#merge_inner                    Merge inner dictionaries of 2 dictionaries.  Outer dictionary lengths must be equal.
#delete_outer_key               Equivalent to deleting a row or a column.
#delete_inner_key               Equivalent to deleting a row or a column.
#insert_outer_inner_header_handle Validate size/format dict header and insert object header.   
#core_insert_outer              Insert a single inner dictionary as {0:x, 1:y, ...} at specified index without safeguards.
#insert_outer                   Insert a single inner dictionary at specified index with safeguards and header handling.
#append_outer                   Append an inner dictionary to a sparse dict in last position.
#core_insert_inner              Insert an entry into all inner dictionaries at specified index without safeguards.
#insert_inner                   Insert an entry into all inner dictionaries at specified index with safeguards and header handling.
#append_inner                   Append an entry into all inner dictionaries in the last position.
#split_outer                    Split before user-specified outer index
#split_inner                    Split before user-specified inner index
#multi_select_outer             Build sparse dict from user-specified outer indices of given sparse dict.
#core_multi_select_inner        Build sparse dict from user-specified inner indices of given sparse dict without safeguards.
#multi_select_inner             Build sparse dict from user-specified inner indices of given sparse dict with safeguards.
# END CREATION, HANDLING & MAINTENANCE ##################################################################################################
# ABOUT ###################################################################################################################################




# MISC ##############################################
module_name                     Return file name.
# END MISC ##########################################


display                         Print sparse dict to screen.
core_find_constants             Finds a column of constants. Returns dict/empty dict of non-zero constant indices, list/empty list of zero idxs.
find_constants                  Finds a column of constants with safeguards. Returns dict/empty dict of non-zero constant indices, list/empty list of zero idxs.



# END ABOUT #############################################################################################################################




    Take list-type of list-types, or dataframe, or datadict {'a':[]} , convert to a dictionary of sparse dictionaries.
    SparseDict converts data to dictionaries that hold non-zero values as values, with index positions as keys.
    e.g. [[0 0 1 0 3 0 1]] is {0: {2:1, 4:3, 6:1} }
    Always create posn for last entry, so that original length of list is retained.  I.e., if the last entry
    of a list is 0, put it in the dict anyway to placehold the original length of the list.
    Always create a dictionary for every list.  [[0,0,1],[0,0,0],[1,1,0]] looks like { 0:{2:1}, 1:[2:0}, 2:{0:1, 1:1, 2:0} }
    The chances of an entire list (whether it is meant to be a column or a row) being all zeros is small, meaning
    there would be little gained by dropping such rows, but the original dimensionality could be lost.
'''

def module_name():
    '''Return file name.'''
    return get_module_name(str(sys.modules[__name__]))



#########################################################################################################################################
#########################################################################################################################################
# CREATION, HANDLING & MAINTENANCE ######################################################################################################








# CURRENTLY ONLY CALLED BY resize()
def resize_inner(DICT1, new_inner_len, calling_fxn=None, HEADER=None):   # LAST IDX IS ALWAYS len()-1, DUE TO ZERO INDEX
    '''Resize sparse dict to user-entered inner dict length.  Reducing size may truncate non-zero values;
        increasing size will introduce zeros (empties) and original inner size placeholder rule (entry for last item even if 0) holds.'''
    fxn = inspect.stack()[0][3]
    DICT1 = dict_init(DICT1, fxn)
    insufficient_dict_args_1(DICT1, fxn)
    non_int(new_inner_len, fxn, "new_inner_len")

    while True:
        if calling_fxn == 'DUMMY PLACEHOLDER':   # ALLOW USER SHORT CIRCUIT IN PROCESS.... NOT IN USE
            if vui.validate_user_str(f'\nReally proceed with inner dict resize?  Non-zero data will might be lost (y/n) > ', 'YN') == 'N':
                break

        DICT1 = clean(DICT1)

        old_inner_len = inner_len(DICT1)

        is_empty = True in [np.array_equiv(HEADER, _) for _ in [ [[]], None ] ]

        if new_inner_len == old_inner_len:  # NEW INNER LEN IS SAME AS OLD, DO NOTHING
            pass

        elif new_inner_len > old_inner_len:
            # DELETE OLD PLACEHOLDERS (key = old inner len - 1 and value == 0, if value != 0 then not a placeholder, dont delete)
            # PUT NEW PLACEHOLDER AT new_len_inner - 1
            for outer_key in DICT1:
                if DICT1[outer_key][old_inner_len - 1] == 0: del DICT1[outer_key][old_inner_len - 1]
                DICT1[int(new_inner_len-1)] = 0

            if not is_empty:
                for inner_key in range(old_inner_len, new_inner_len):
                    HEADER[0].append(inner_key)

        elif new_inner_len < old_inner_len:
            # DELETE ANYTHING AFTER old_inner_len - 1, PUT NEW PLACEHOLDERS as new_inner_len - 1 if NEEDED
            for outer_key in DICT1:
                for inner_key in range(new_inner_len, old_inner_len):
                    if inner_key in DICT1[outer_key]: del DICT1[outer_key][inner_key]
                if new_inner_len - 1 not in DICT1[outer_key]:
                    DICT1[int(outer_key)][int(new_inner_len-1)] = 0

            if not is_empty:
                for inner_key in range(new_inner_len, old_inner_len):
                    HEADER[0].pop(inner_key)

        break

    return DICT1, HEADER


# CURRENTLY ONLY CALLED BY resize()
def resize_outer(DICT1, new_outer_len, calling_fxn=None, HEADER=None):   # LAST IDX IS ALWAYS len()-1, DUE TO ZERO INDEX
    '''Resize sparse dict to user-entered outer dict length.  Reducing size may truncate non-zero values;
        increasing size will introduce zeros (placeholder inner dicts) and original outer size placeholder rule holds.'''

    fxn = inspect.stack()[0][3]
    DICT1 = dict_init(DICT1, fxn)
    insufficient_dict_args_1(DICT1, fxn)
    non_int(new_outer_len, fxn, "new_outer_len")

    while True:
        if calling_fxn == 'DUMMY_PLACEHOLDER':   # ALLOW USER SHORT CIRCUIT IN PROCESS.... NOT IN USE
            if vui.validate_user_str(f'\nReally proceed with outer dict resize?  Non-zero data will might be lost (y/n) > ', 'YN') == 'N':
                break

        DICT1 = clean(DICT1)

        old_outer_len = outer_len(DICT1)

        is_empty = True in [np.array_equiv(HEADER, _) for _ in [ [[]], None ] ]

        if new_outer_len == old_outer_len:    # NEW INNER LEN IS SAME AS OLD, DO NOTHING
            pass

        elif new_outer_len > old_outer_len:
            # PUT PLACEHOLDERS IN THE NEW KEYS
            for outer_key in range(old_outer_len, new_outer_len):
                DICT1[int(outer_key)] = {int(inner_len(DICT1)): 0}
                if not is_empty: HEADER[0].append(outer_key)

        elif new_outer_len < old_outer_len:
            for outer_key in range(new_outer_len, old_outer_len):
                del DICT1[outer_key]
                if not is_empty: HEADER[0].pop(outer_key)

        break

    return DICT1, HEADER


def resize(DICT1, len_outer_key, len_inner_key, HEADER=None, header_goes_on=None):  # LAST OUTER AND INNER IDXS ARE ALWAYS len()-1, DUE TO ZERO INDEXING
    '''Resize sparse dict to user-entered (len outer dict, len inner dicts) dimensions.  Reducing size may truncate non-zero values;
        increasing size will introduce zeros (empties in inner dicts, placeholder in outer dict) and original size placeholder rules hold.'''

    fxn = inspect.stack()[0][3]
    DICT1 = dict_init(DICT1, fxn)
    insufficient_dict_args_1(DICT1, fxn)
    non_int(len_outer_key, fxn, "len_outer_key")
    non_int(len_inner_key, fxn, "len_inner_key")

    if isinstance(header_goes_on, str): header_goes_on = header_goes_on.upper()

    if header_goes_on == 'OUTER':
        DICT1, HEADER = resize_outer(DICT1, len_outer_key, calling_fxn=fxn, HEADER=HEADER)
        DICT1, DUM = resize_inner(DICT1, len_inner_key, calling_fxn=fxn, HEADER=HEADER)
    elif header_goes_on == 'INNER':
        DICT1, DUM = resize_outer(DICT1, len_outer_key, calling_fxn=fxn, HEADER=HEADER)
        DICT1, HEADER = resize_inner(DICT1, len_inner_key, calling_fxn=fxn, HEADER=HEADER)
    elif header_goes_on is None:
        DICT1, DUM = resize_outer(DICT1, len_outer_key, calling_fxn=fxn, HEADER=HEADER)
        DICT1, HEADER = resize_inner(DICT1, len_inner_key, calling_fxn=fxn, HEADER=HEADER)
    else:
        raise ValueError(f'INVALID header_goes_on IN {module_name()}.{fxn}().  MUST BE "outer" or "inner".')

    return DICT1, HEADER





def merge_outer(DICT1, DICT2, HEADER1=None, HEADER2=None):
    '''Merge outer dictionaries of 2 dictionaries with safeguards.  Inner dictionary lengths must be equal.'''

    fxn = inspect.stack()[0][3]

    if DICT1 == {} and DICT2 == {}:         # IF DICT1 AND DICT2 ARE EMPTY
        insufficient_dict_args_2(DICT1, DICT2, fxn)
    elif not DICT1 == {} and DICT2 == {}:    # IF DICT2 IS EMPTY
        DICT1 = dict_init(DICT1, fxn)
        insufficient_dict_args_1(DICT1, fxn)
        DICT1 = clean(DICT1)
        return DICT1, HEADER1
    elif not DICT2 == {} and DICT1 == {}:    # IF DICT1 IS EMPTY
        DICT2 = dict_init(DICT2, fxn)
        insufficient_dict_args_1(DICT2, fxn)
        DICT2 = clean(DICT2)
        return DICT2, HEADER2
    else:
        DICT1 = dict_init(DICT1, fxn)
        DICT2 = dict_init(DICT2, fxn)
        insufficient_dict_args_2(DICT1, DICT2, fxn)
        inner_len_check(DICT1, DICT2, fxn)

        DICT1 = clean(DICT1)
        DICT2 = clean(DICT2)

        return core_merge_outer(DICT1, DICT2, HEADER1=HEADER1, HEADER2=HEADER2)


def core_merge_outer(DICT1, DICT2, HEADER1=None, HEADER2=None):
    '''Merge outer dictionaries of 2 dictionaries without safeguards.  Inner dictionary lengths must be equal.'''

    if not DICT1 == {} and DICT2 == {}:    # IF DICT2 IS EMPTY
        return DICT1, HEADER1
    elif not DICT2 == {} and DICT1 == {}:    # IF DICT1 IS EMPTY
        return DICT2, HEADER2
    else:
        # CANT JUST MERGE THE 2 DICTS, THEY MIGHT (PROBABLY) HAVE MATCHING OUTER KEYS AND OVERWRITE
        # GET outer_len of DICT1 TO KNOW HOW TO INDEX DICT2, REINDEX DICT2 ON THE FLY

        NEW_DICT2_KEYS = np.fromiter(DICT2.keys(), dtype=np.int16) + outer_len(DICT1)
        # PIZZA 24_05_07_08_30_00 THIS map int IS A PATCH TO FIX A NON-INT KEY PROBLEM
        NEW_DICT2_KEYS = list(map(int, NEW_DICT2_KEYS))

        NEW_DICT = DICT1 | dict((zip(NEW_DICT2_KEYS, DICT2.values())))
        del NEW_DICT2_KEYS

        if not True in [np.array_equiv(HEADER1, _) for _ in [[[]], None]] and \
                not True in [np.array_equiv(HEADER2, _) for _ in [[[]], None]]:
            HEADER1 = np.array([*HEADER1[0], *HEADER2[0]], dtype='<U500')
            HEADER1.reshape((1, len(HEADER1)))

        return NEW_DICT, HEADER1


def merge_inner(DICT1, DICT2, HEADER1=None, HEADER2=None):
    '''Merge inner dictionaries of 2 dictionaries.  Outer dictionary lengths must be equal.'''
    fxn = inspect.stack()[0][3]
    DICT1 = dict_init(DICT1, fxn)
    DICT2 = dict_init(DICT2, fxn)
    insufficient_dict_args_2(DICT1, DICT2, fxn)
    DICT1 = clean(DICT1)
    DICT2 = clean(DICT2)
    outer_len_check(DICT1, DICT2, fxn)

    # CANT JUST MERGE THE 2 DICTS, THEY MIGHT (PROBABLY) HAVE MATCHING INNER KEYS AND OVERWRITE
    # GET inner_len of DICT1 TO KNOW HOW TO INDEX DICT2, REINDEX DICT2 ON THE FLY
    _inner_len1 = inner_len(DICT1)
    _inner_len2 = inner_len(DICT2)
    combined_inner_len = _inner_len1 + _inner_len2
    pseudo_dict2_outer_key = 0
    for outer_key in DICT2.keys():   # DICT1 outer len must == DICT 2 outer len
        # CURRENTLY UNABLE TO CLEAN DICT2, SO IF OUTER KEYS NOT CONTIGUOUS, USE PSEUDOKEY TO MATCH AGAINST DICT1
        # CHECK TO SEE IF VALUE AT END OF DICT1 INNER IS 0, IF SO, DELETE
        if DICT1[pseudo_dict2_outer_key][_inner_len1-1] == 0: del DICT1[pseudo_dict2_outer_key][_inner_len1-1]
        for inner_key in DICT2[outer_key]:
            DICT1[int(pseudo_dict2_outer_key)][int(_inner_len1 + inner_key)] = DICT2[outer_key][inner_key]
        else: # WHEN GET TO LAST INNER KEY, ENFORCE PLACEHOLDING RULES
            DICT1[int(pseudo_dict2_outer_key)][int(combined_inner_len - 1)] = \
                DICT1[pseudo_dict2_outer_key].get(combined_inner_len - 1, 0)

        pseudo_dict2_outer_key += 1

    if not True in [np.array_equiv(HEADER1, _) for _ in [[[]], None]] and \
            not True in [np.array_equiv(HEADER2, _) for _ in [[[]], None]]:
        HEADER1 = [[*HEADER1[0], *HEADER2[0]]]

    return DICT1, HEADER1


def delete_outer_key(DICT, OUTER_KEYS_TO_DELETE_AS_LIST, HEADER=None):
    '''Equivalent to deleting a row or a column.'''

    DICT1 = deepcopy(DICT)   # TO PREVENT BLOWBACK TO ORIGINAL OBJECT, DEMONSTRATED TO BE HAPPENING 10/15/22

    fxn = inspect.stack()[0][3]
    insufficient_dict_args_1(DICT1, fxn)
    insufficient_list_args_1(OUTER_KEYS_TO_DELETE_AS_LIST, fxn)
    for delete_key in OUTER_KEYS_TO_DELETE_AS_LIST:
        non_int(delete_key, fxn, "key")

    DICT1 = clean(DICT1)
    _outer_len = outer_len(DICT1)

    _min_delete_key = min(OUTER_KEYS_TO_DELETE_AS_LIST)
    _max_delete_key = max(OUTER_KEYS_TO_DELETE_AS_LIST)

    if _min_delete_key < 0:
        raise Exception(f'Outer key {_min_delete_key} out of bounds for {module_name()}.{fxn}(). Must be >= 0.')
    if _max_delete_key > _outer_len - 1:
        raise Exception(f'Outer key {_max_delete_key} out of bounds for {module_name()}.{fxn}(). Must be <= {_outer_len-1}.')

    outer_key_adjustment = 0
    for outer_key in range(_min_delete_key, _outer_len):   # MUST ITERATE OVER ALL KEYS AFTER LOWEST, TO CAPTURE CORRECT AMOUNT TO SUBTRACT
        if outer_key in OUTER_KEYS_TO_DELETE_AS_LIST:
            del DICT1[outer_key]
            outer_key_adjustment += 1
        if outer_key not in OUTER_KEYS_TO_DELETE_AS_LIST:
            DICT1[int(outer_key - outer_key_adjustment)] = DICT1.pop(outer_key)

        if not True in [np.array_equiv(HEADER, _) for _ in [ [[]], None ] ]:
            HEADER[0].pop(delete_key)

    return DICT1, HEADER


def delete_inner_key(DICT, INNER_KEYS_TO_DELETE_AS_LIST, HEADER=None):
    '''Equivalent to deleting a row or a column.'''

    DICT1 = deepcopy(DICT)   # TO PREVENT BLOWBACK TO ORIGINAL OBJECT, DEMONSTRATED TO BE HAPPENING 10/15/22

    fxn = inspect.stack()[0][3]
    DICT1 = dict_init(DICT1, fxn)
    insufficient_dict_args_1(DICT1, fxn)
    insufficient_list_args_1(INNER_KEYS_TO_DELETE_AS_LIST, fxn)
    for delete_key in INNER_KEYS_TO_DELETE_AS_LIST:
        non_int(delete_key, fxn, "key")

    DICT1 = clean(DICT1)
    _inner_len = inner_len(DICT1)

    _min_delete_key = min(INNER_KEYS_TO_DELETE_AS_LIST)
    _max_delete_key = max(INNER_KEYS_TO_DELETE_AS_LIST)

    if _min_delete_key < 0:
        raise Exception(f'Inner key {_min_delete_key} out of bounds for {module_name()}.{fxn}(). Must be >= 0.')
    if _max_delete_key > _inner_len - 1:
        raise Exception(f'Inner key {_max_delete_key} out of bounds for {module_name()}.{fxn}(). Must be <= {_inner_len - 1}.')

    # MUST REDUCE ALL keys AFTER delete_key BY 1!!! DONT JUST SET delete_key TO ZERO BY JUST DELETING IT!!!

    for outer_key in DICT1.copy():    # ONLY COPY OUTER THINGS
        inner_key_adjustment = 0
        for inner_key in range(_min_delete_key, _inner_len):
            if inner_key in INNER_KEYS_TO_DELETE_AS_LIST:
                inner_key_adjustment += 1
            if inner_key in DICT1[outer_key] and inner_key in INNER_KEYS_TO_DELETE_AS_LIST:
                del DICT1[outer_key][inner_key]
            if inner_key in DICT1[outer_key] and inner_key not in INNER_KEYS_TO_DELETE_AS_LIST:
                DICT1[int(outer_key)][int(inner_key - inner_key_adjustment)] = DICT1[outer_key].pop(inner_key)

            if inner_key == 0 and not True in [np.array_equiv(HEADER, _) for _ in [ [[]], None ] ]:  # DO THIS ON ONLY ONE PASS
                HEADER[0].pop(inner_key)

        if DICT1[outer_key] == {}:
            del DICT1[outer_key]

        else:
            # WHEN DELETING THE LAST inner_key IN INNER DICT, MANAGE PLACEHOLDER RULES
            if _inner_len - 1 - inner_key_adjustment not in DICT1[outer_key]:
                DICT1[int(outer_key)][int(inner_key - inner_key_adjustment)] = 0

    # REORDER INNER IDXS
    if DICT1 != {}:
        for inner_key in sorted(DICT1[outer_key].copy()):
            DICT1[outer_key][inner_key] = DICT1[outer_key].pop(inner_key)

    return DICT1, HEADER


def insert_outer_inner_header_handle(DICT_HEADER1, INSERT_HEADER, dict1_outer_len, dict1_inner_len, header_axis, active_axis,
                                     ins_len, fxn):
    '''Validate size/format of insert object header and receiving object header.'''

    # TEST IS INCIDENTAL WITH insert_outer AND insert_inner MODULES.

    __ = f'sparse_dict.{fxn}()'

    # IF DICT1_HEADER OR INSERT_HEADER ARE NOT None, header_axis MUST BE PROVIDED AND MUST BE 0 OR 1
    if (not DICT_HEADER1 is None or not INSERT_HEADER is None):
        if header_axis not in [0, 1]:
            raise ValueError(f'IF ONE OR BOTH HEADER OBJECTS ARE GIVEN, header_axis MUST BE PROVIDED AND MUST BE 0 OR 1.')
        if active_axis not in [0, 1]:
            raise ValueError(f'active_axis MUST BE PROVIDED AND MUST BE 0 OR 1.')

    # ENSURE DICT_HEADER IS LIST-TYPE AND CONVERT TO np.[[]]
    if not DICT_HEADER1 is None:
        if not isinstance(DICT_HEADER1, (list, tuple, np.ndarray)):
            raise TypeError(f'{__} DICT_HEADER1 MUST BE A LIST-TYPE ENTERED AS [] OR [[]]')
        else: DICT_HEADER1 = np.array(DICT_HEADER1, dtype='<U200').reshape((1,-1))

        # CHECK IF DICT_HEADER1 MATCHES LEN OF ASSIGNED AXIS ( 0 --> len_outer == len(DICT1_HEADER), 1 --> len_inner == len(DICT1_HEADER) )
        _text = lambda axis: f'{__}: DICT_HEADER1 LENGTH MUST MATCH {"OUTER" if axis==0 else "INNER"} LENGTH OF GIVEN DICT WHEN ' \
                                f'header_axis IS {axis}. MAYBE PROVIDING HEADER WHEN NOT NEEDED, OR SPECIFYING WRONG AXIS?'
        if header_axis == 0 and len(DICT_HEADER1[0]) != dict1_outer_len: raise ValueError(_text(0))
        elif header_axis == 1 and len(DICT_HEADER1[0]) != dict1_inner_len: raise ValueError(_text(1))
        del _text

    # ENSURE INSERT_HEADER IS LIST-TYPE AND CONVERT TO np.[[]]
    if not INSERT_HEADER is None:
        if not isinstance(INSERT_HEADER, (list, tuple, np.ndarray)):
            raise TypeError(f'{__} INSERT_HEADER MUST BE A LIST-TYPE ENTERED AS [] OR [[]]')
        else: INSERT_HEADER = np.array(INSERT_HEADER, dtype='<U200').reshape((1, -1))

        # CHECK IF INS_HEADER MATCHES LEN OF INS_OBJ
        if len(INSERT_HEADER[0]) != ins_len: raise ValueError(f'{__}: INS_HEADER LENGTH MUST MATCH INS_OBJ LENGTH.')

    # IF INSERT_HEADER IS PROVIDED BUT NOT DICT_HEADER1, MAKE DUMMY DICT_HEADER1
    if not INSERT_HEADER is None and DICT_HEADER1 is None:
        warnings.warn(f'{__}: HEADER OF INSERTED OBJECT WAS PROVIDED AND HEADER OF RECEIVING OBJECT WAS NOT.')
        if header_axis==0: DICT_HEADER1 = np.fromiter((f'DICT1_COL_{idx+1}' for idx in range(dict1_outer_len)), dtype='<U20').reshape((1,-1))
        elif header_axis == 1: DICT_HEADER1 = np.fromiter((f'DICT1_COL_{idx+1}' for idx in range(dict1_inner_len)), dtype='<U20').reshape((1,-1))

    # IF DICT_HEADER1 WAS PROVIDED BUT NOT INSERT_HEADER, MAKE DUMMY INSERT_HEADER
    if INSERT_HEADER is None and not DICT_HEADER1 is None:
        warnings.warn(f'{__}: HEADER OF RECEIVING OBJECT WAS PROVIDED AND HEADER OF INSERTED OBJECT WAS NOT.')
        if header_axis != active_axis: INSERT_HEADER = [[]]
        elif header_axis == active_axis: INSERT_HEADER = np.fromiter((f'INS_COL_{idx+1}' for idx in range(ins_len)), dtype='<U20').reshape((1,-1))

    if DICT_HEADER1 is None and INSERT_HEADER is None:
        pass
        # DONT BUILD ANY HEADERS IF NONE WERE PASSED

    return DICT_HEADER1, INSERT_HEADER


def core_insert_outer(INPUT_DICT, index, INS_OBJ):
    '''Insert a single inner dictionary as {0:x, 1:y, ...} at specified index without safeguards.'''

    # TEST CODE IS LOCATED IN sparse_dict_test_modules.insert_outer__function_test().

    # 10/11/22 IN SITUATION WHERE DOING y = sd.(core)_insert(append)_outer(x,....) IN ANOTHER MODULE, EVEN THO RETURNING
    # AS y, THE INPUT x IS BEING MODIFIED AS WELL :(.  PRINTING x IN OTHER MODULE AFTER RETURNING y SHOWED x HAD CHANGED.
    # PUTTING deepcopy HERE BREAKS THAT CHAIN (AND BLOWS UP MEMORY :( )

    DICT1 = deepcopy(INPUT_DICT)
    del INPUT_DICT


    if isinstance(INS_OBJ, dict):
        if is_sparse_inner(INS_OBJ):
            INS_OBJ = {0: INS_OBJ}
        if len(INS_OBJ)==1: INS_OBJ[0] = INS_OBJ.pop(list(INS_OBJ.keys())[0])
    elif isinstance(INS_OBJ, (np.ndarray, list, tuple)):
        INS_OBJ = np.array(INS_OBJ)
        if len(INS_OBJ.shape)==1: INS_OBJ = INS_OBJ.reshape((1, -1))
        if 'INT' in str(INS_OBJ.dtype).upper(): INS_OBJ = zip_list_as_py_int(INS_OBJ)
        elif 'FLOAT' in str(INS_OBJ.dtype).upper(): INS_OBJ = zip_list_as_py_float(INS_OBJ)
    else: raise TypeError(f'\n{module_name()}.core_insert_outer() INVALID OBJECT TYPE ({type(INS_OBJ)}) PASSED; '
                                    f'MUST BE LIST-TYPE OR SPARSE DICTIONARY.')

    if DICT1 is None or DICT1 == {}:  # IF NO RECEIVING DICT WAS PASSED, THEN THE INSERT SIMPLY BECOMES THE NEW DICT
        return INS_OBJ


    dict1_outer_len, dict1_inner_len = shape_(DICT1)
    ins_obj_outer_len, ins_obj_inner_len = shape_(INS_OBJ)

    # CHECK IF INSERTION IDX IS IN RANGE # ############################################
    if index > dict1_outer_len or index < 0:
        raise ValueError(f'\n{module_name()}.core_insert_outer() OUTER INSERT INDEX {index} OUT OF RANGE FOR SPARSE DICT OF '
                                  f'OUTER LEN {dict1_outer_len:,.0f}')
    #### VALIDATE INSERT AND DICT LENS #############################################################################################

    # SEE WHAT ORIENTATION OF INS_OBJ MATCHES UP AGAINST INNER LEN OF DICT1
    if ins_obj_inner_len==dict1_inner_len: pass  # INS_OBJ WAS PASSED ORIENTED THE SAME AS DICT1
    elif ins_obj_outer_len==dict1_inner_len:   # INS_OBJ WAS PASSED ORIENTED AS ROW, CHANGE TO MATCH DICT1
        INS_OBJ = core_sparse_transpose(INS_OBJ)
    else:   # NONE OF THE DIMENSIONS OF INS_OBJ MATCH THE INNER LEN OF DICT1
        raise ValueError(f'\n{module_name()}.core_insert_outer() INS_OBJ WITH DIMENSIONS {shape_(INS_OBJ)} CANNOT '
                        f'BE FORCED TO FIT A RECEIVING DICT WITH INNER LEN OF {dict1_inner_len}.')

    # GET SHAPE OF INS_OBJ AGAIN AFTER ANY REORIENTING THAT MAY HAVE HAPPENED
    ins_obj_outer_len, ins_obj_inner_len = shape_(INS_OBJ)

    if DICT1 != {} and ins_obj_inner_len != dict1_inner_len:  # ALLOWS APPEND TO {}
        raise ValueError(f'\n{module_name()}.core_insert_outer() '
            f'LENGTH OF INSERTED OBJECT ({ins_obj_inner_len:,.0f}) DOES NOT EQUAL INNER LENGTH OF RECEIVING OBJECT ({dict1_inner_len:,.0f})')

    # APPENDING
    if index == dict1_outer_len:
        for col_idx in set(INS_OBJ.keys()):
            DICT1[int(dict1_outer_len + col_idx)] = INS_OBJ.pop(col_idx)
    # INSERTING
    else:
        DICT2 = {}
        for dict1_outer_key in range(index, dict1_outer_len):
            DICT2[int(dict1_outer_key + ins_obj_outer_len)] = DICT1.pop(dict1_outer_key)

        # INCREMENT INS_OBJs OUTER IDXS
        INS_OBJ = dict((zip(np.add(np.fromiter(INS_OBJ.keys(), dtype=np.int32), index).tolist(), INS_OBJ.values())))

        DICT1 = DICT1 | INS_OBJ | DICT2

        del INS_OBJ, DICT2

    return DICT1


def insert_outer(DICT1, index, LIST_OR_DICT_TO_INSERT, DICT_HEADER1=None, INSERT_HEADER=None,
                 header_axis=None, fxn=None):
    '''Insert a single inner dictionary at specified index with safeguards and header handling.'''

    # TEST CODE IS LOCATED IN sparse_dict_test_modules.insert_outer__function_test().

    # ASSUMES DICT1 IS "CLEAN"
    fxn = inspect.stack()[0][3] if fxn is None else fxn
    DICT1 = dict_init(DICT1, fxn)
    insufficient_dict_args_1(DICT1, fxn)
    dict1_outer_len = outer_len(DICT1)
    dict1_inner_len = inner_len(DICT1)

    DICT1 = core_insert_outer(DICT1, index, LIST_OR_DICT_TO_INSERT)

    del LIST_OR_DICT_TO_INSERT

    # IF INS_HEADER WAS NOT PASSED, MEASURE THE DELTA OF DICT1 BEFORE & AFTER INSERT TO FIND
    # OUT HOW LONG INS_OBJ WAS. USE THIS DELTA TO CREATE THE DUMMY HEADER.
    DICT_HEADER1, INSERT_HEADER = insert_outer_inner_header_handle(DICT_HEADER1, INSERT_HEADER, dict1_outer_len,
                                   dict1_inner_len, header_axis, 0, outer_len(DICT1) - dict1_outer_len, fxn)

    if not DICT_HEADER1 is None and not INSERT_HEADER is None:
        if header_axis == 0:
            # BECAUSE outer AND HEADER IS TIED TO OUTER (AXIS 0), MODIFTY HEADER
            DICT_HEADER1 = np.hstack((DICT_HEADER1[...,:index], INSERT_HEADER, DICT_HEADER1[..., index:])).astype('<U200')
        # elif header_axis == 1: pass
            # NO CHANGE TO DICT_HEADER1, JUST RETURN THE ORIGINAL GIVEN OR WHAT CAME OUT OF header_handle

    if not DICT_HEADER1 is None: return DICT1, DICT_HEADER1
    else: return DICT1


def append_outer(DICT1, LIST_OR_DICT_TO_INSERT, DICT_HEADER1=None, INSERT_HEADER=None, header_axis=None, fxn=None):
    '''Append an inner dictionary to a sparse dict in last position.'''

    # TEST CODE IS LOCATED IN sparse_dict_test_modules.insert_outer__function_test().

    fxn = inspect.stack()[0][3] if fxn is None else fxn
    # VALIDATION IS HANDLED IN insert_outer
    index = len(DICT1)

    if not (DICT_HEADER1 is None and INSERT_HEADER is None):
        DICT1, DICT_HEADER1 = insert_outer(DICT1, index, LIST_OR_DICT_TO_INSERT,
                                DICT_HEADER1=DICT_HEADER1, INSERT_HEADER=INSERT_HEADER, header_axis=header_axis, fxn=fxn)
        return DICT1, DICT_HEADER1
    else:
        DICT1 = insert_outer(DICT1, index, LIST_OR_DICT_TO_INSERT,
                                DICT_HEADER1=DICT_HEADER1, INSERT_HEADER=INSERT_HEADER, header_axis=header_axis, fxn=fxn)
        return DICT1


def core_insert_inner(INPUT_DICT, insert_index, INS_OBJ):
    ''' Insert an entry into all inner dictionaries at specified index without safeguards.'''

    # TEST CODE IS LOCATED IN sparse_dict_test_modules.insert_inner__function_test().

    ''' OUTLINE
    I) GET shape OF INS_OBJ, MAKE SURE IS STANDARDIZED TO [[]] AND {0: {}}
    II) VALIDATE / GET SHAPE OF DICT1
        A) DICT1 NOT GIVEN --- FINAL DICT DIMENSIONS MUST BE SUP_OBJ DIMENSIONS AS GIVEN. IF SHAPE OF INS_OBJ IS (X,Y), Y 
            DETERMINES # OUTER DICTS
            1) CONSTRUCT DICT1 AS EMPTY INNER DICTS, FILLING outer_len
            2) GET SHAPE OF DICT1  (MUST BE (outer_len,0)
        B) DICT1 IS GIVEN --- GET SHAPE OF DICT1
    III) VALIDATE / STANDARDIZE INS_OBJ
        A) DICT1 IS NOT GIVEN --- DONT WORRY ABOUT INS_OBJ SHAPE, JUST GET FROM [[INNER VALUES FOR inner_idx insert_index], 
            [INNER VALUES FOR inner_idx insert_index+1], ...] 
            INTO SD FORMAT WITH [[INNER VALUES FOR outer_idx 0], [INNER VALUES FOR  outer_idx 1], ...]
        B) DICT1 IS GIVEN --- MUST HAVE SHAPE OF DICT1
            1) IF (X,Y) IS SHAPE OF INS_OBJ (COULD BE NP OR SD), ASSUME Y WAS INTENDED TO ALIGN TO outer_len OF DICT1, MAKING
                X THE NUMBER OF VALUES TO BE INSERTED IN EACH INNER DICT
            2) IF DOES NOT ALIGN, CHECK IF X ALIGNS TO outer_len OF DICT
        C) MUST GET INS_OBJ AS SD & ORIENTED AS [[INNER VALUES outer_idx 0], [INNER VALUES outer_idx 1],...]
    IV) INCREMENT INS_OBJ INNER IDXS TO START AT insert_index
    V) COMBINE DICT1 AND INS_OBJ
        A) IF INSERTING AT END (APPENDING)
            1) REMOVE PLACEHOLDERS FROM INPUT_DICT
            2) MERGE DICT1 & INS_OBJ USING |
            3) APPLY PLACEHOLDERS
            4) TERMINATE MODULE BY RETURNING DICT1
        B) IF INSERTING INS_OBJ SOMEWHERE IN THE MIDDLE
            1) SPLIT DICT1[outer_idx] AT insert_index INTO 2 DICTS (LEFT_DICT & RIGHT_DICT)
            2) INCREMENT RIGHT_DICT INNER IDXS BY (number of inner idxs IN SUP_OBJ)
            3) MERGE ALL DICTS TOGETHER USING LEFT_DICT[outer_idx] | INS_OBJ[outer_idx] | RIGHT_DICT[outer_idx]
            4) TERMINATE MODULE BY RETURNING DICT1        
    '''

    # 10/11/22 IN SITUATION WHERE DOING y = sd.(core)_insert(append)_inner(x,....) IN ANOTHER MODULE, EVEN THO RETURNING
    # AS y, THE INPUT x IS BEING MODIFIED AS WELL :(.  PRINTING x IN OTHER MODULE AFTER RETURNING y SHOWED x HAD CHANGED.
    # PUTTING deepcopy HERE BREAKS THAT CHAIN (AND BLOWS UP MEMORY :( )
    DICT1 = deepcopy(INPUT_DICT)
    del INPUT_DICT

    # I) GET shape OF INS_OBJ, MAKE SURE IS STANDARDIZED TO [[]] AND {0: {}}
    # ASSUME INS_OBJ IS NP OR SD, AND USER GAVE IT AS 'ROW'  (3/25/23 THINKING "ROW" OR "COLUMN" DOESNT MATTER, AS LONG AS DICT & INS ARE LABELED THE SAME)

    # CHECK INSERT OBJ IS NOT EMPTY, VERIFY IS LIST-TYPE OR DICT
    empty = False
    if INS_OBJ is None: empty = True
    elif isinstance(INS_OBJ, (list, tuple, np.ndarray)):
        if np.array(INS_OBJ).size == 0: empty = True
    elif isinstance(INS_OBJ, dict):
        if INS_OBJ == {}: empty = True
    else: raise TypeError(f'\n{module_name()}.core_insert_inner() INVALID INSERTION OBJECT TYPE {type(INS_OBJ)}')
    if empty: raise ValueError(f'\n{module_name()}.core_insert_inner() INSERT OBJECT IS EMPTY')

    if is_sparse_inner(INS_OBJ): INS_OBJ = {0: INS_OBJ}
    elif isinstance(INS_OBJ, (np.ndarray, list, tuple)):
        INS_OBJ = np.array(INS_OBJ)
        if len(INS_OBJ.shape)==1: INS_OBJ=INS_OBJ.reshape((1,-1))
    elif is_sparse_outer(INS_OBJ): pass

    ins_shape = gs.get_shape('INS_OBJ', INS_OBJ, 'ROW')

    # II) VALIDATE / GET SHAPE OF DICT1
    # A) DICT1 NOT GIVEN --- FINAL DICT DIMENSIONS MUST BE SUP_OBJ DIMENSIONS AS GIVEN. IF SHAPE OF INS_OBJ IS (X,Y), Y DETERMINES # OUTER DICTS
    if DICT1 == {} or DICT1 is None:
        # 1) CONSTRUCT DICT1 AS EMPTY INNER DICTS, FILLING outer_len
        DICT1 = {int(outer_idx): {} for outer_idx in range(ins_shape[1])}
        # 2) GET SHAPE OF DICT1  (MUST BE (outer_len,0)
        dict_shape = gs.get_shape('DICT1', DICT1, 'ROW')
    # B) DICT1 IS GIVEN --- GET SHAPE OF DICT1
    else: dict_shape = gs.get_shape('DICT1', DICT1, 'ROW')

    # III) VALIDATE / STANDARDIZE INS_OBJ #################################################################################
    # A) DICT1 IS NOT GIVEN --- DONT WORRY ABOUT INS_OBJ SHAPE, JUST GET FROM [[INNER VALUES FOR inner_idx insert_index],
    #         [INNER VALUES FOR inner_idx insert_index+1], ...]
    #         INTO SD FORMAT WITH [[INNER VALUES FOR outer_idx 0], [INNER VALUES FOR  outer_idx 1], ...]
    if 0 in dict_shape:   # MUST HAVE BEEN SET TO THIS ABOVE IF EMPTY INPUT_DICT WAS PASSED, TRANSPOSE & HANDLE CONVERTING TO SD BELOW
        if isinstance(INS_OBJ, dict): INS_OBJ = core_sparse_transpose(INS_OBJ)
        elif isinstance(INS_OBJ, np.ndarray): INS_OBJ = INS_OBJ.transpose()

    # B) DICT1 IS GIVEN --- MUST HAVE SHAPE OF DICT1
    else:
        if insert_index not in range(dict_shape[1] + 1):
            raise ValueError(f'insert_idx ({insert_index}) IS OUTSIDE OF RANGE OF PASSED INPUT OBJECT')

        # 1) IF (X,Y) IS SHAPE OF INS_OBJ (COULD BE NP OR SD), ASSUME Y WAS INTENDED TO ALIGN TO outer_len OF DICT1, MAKING
        #             X THE NUMBER OF VALUES TO BE INSERTED IN EACH INNER DICT
        if dict_shape[0] == ins_shape[1]: # WAS GIVEN WITH EXPECTED ORIENTATION W-R-T GIVEN DICT, SO CHANGE TO [[]=outer]
            if isinstance(INS_OBJ, dict): INS_OBJ = core_sparse_transpose(INS_OBJ)
            elif isinstance(INS_OBJ, np.ndarray): INS_OBJ = INS_OBJ.transpose()
            else: raise AssertionError(f'OBJECT RE-ORIENTATION TO [[]=OUTER] IS FAILING')
        # 2) IF DOES NOT ALIGN, CHECK IF X ALIGNS TO outer_len OF DICT
        elif dict_shape[0] == ins_shape[0]: pass  # ALREADY IN DESIRED ORIENTATION, HANDLE CONVERSION TO SD BELOW
        else: raise ValueError(f'PASSED INSERT OBJECT DOES NOT ALIGN TO INPUT DICT OUTER LENGTH IN ANY ORIENTATION')

    # C) MUST GET INS_OBJ AS SD & ORIENTED AS [[INNER VALUES outer_idx 0], [INNER VALUES outer_idx 1],...]
        # ORIENTATION SHOULD HAVE BEEN HANDLED ABOVE, SO GET INTO SD
    if isinstance(INS_OBJ, np.ndarray):
        if 'INT' in str(INS_OBJ.dtype).upper(): INS_OBJ = zip_list_as_py_int(INS_OBJ)
        elif 'FLOAT' in str(INS_OBJ.dtype).upper(): INS_OBJ = zip_list_as_py_float(INS_OBJ)

    # RESET SHAPE HERE NOW THAT INS LAYOUT IS STANDARDIZED
    ins_shape = gs.get_shape('INS_OBJ', INS_OBJ, 'ROW')

    # END III) VALIDATE / STANDARDIZE INS_OBJ #################################################################################

    # IV) INCREMENT INS_OBJ INNER IDXS TO START AT insert_index
    for outer_key in INS_OBJ:
        INS_OBJ[int(outer_key)] = dict((
                zip(np.add(np.fromiter(INS_OBJ[outer_key].keys(), dtype=int), insert_index).tolist(), list(INS_OBJ[outer_key].values()))
        ))

    # V) COMBINE DICT1 AND INS_OBJ
    #     A) IF INSERTING AT END (APPENDING)
    if insert_index == dict_shape[1]:
        for outer_idx in DICT1:
    #         1) REMOVE PLACEHOLDERS FROM DICT1
            # 6/6/23 -- IF EMPTY IS BLOWING UP WHEN TRY TO INDEX IN --- GO TO try/except
            try:
                if DICT1[outer_idx][dict_shape[1]-1] == 0: del DICT1[outer_idx][dict_shape[1]-1]
            except: pass
    #         2) MERGE DICT1 & INS_OBJ USING |
            DICT1[int(outer_idx)] = DICT1[outer_idx] | INS_OBJ.pop(outer_idx)
    #         3) APPLY PLACEHOLDERS ---- REMEMBER INNER LEN IS NOW ONE LONGER, SO dict_shape[1] NOT dict_shape[1]-1 !!
            DICT1[int(outer_idx)][int(dict_shape[1])] = DICT1[int(outer_idx)].get(dict_shape[1], 0)
    #         4) TERMINATE MODULE BY RETURNING DICT1
        return DICT1
    else:   # B) IF INSERTING INS_OBJ SOMEWHERE IN THE MIDDLE
    #   1) SPLIT DICT1[outer_idx] AT insert_index INTO 2 DICTS (LEFT_DICT & RIGHT_DICT)

        LEFT_DICT, RIGHT_DICT = {}, {}
        for outer_idx in set(DICT1.keys()):
            ACTV_INNER = DICT1.pop(outer_idx)
            ACTV_KEYS = np.fromiter(ACTV_INNER.keys(), dtype=np.int32)
            ACTV_VALUES = np.fromiter(ACTV_INNER.values(), dtype=np.float64)
            del ACTV_INNER

            LEFT_DICT[int(outer_idx)] = dict((
                zip(ACTV_KEYS[ACTV_KEYS < insert_index].tolist(), ACTV_VALUES[ACTV_KEYS < insert_index].tolist())
            ))

            #   2) INCREMENT RIGHT_DICT INNER IDXS BY (number of inner idxs IN SUP_OBJ)
            RIGHT_DICT[int(outer_idx)] = dict((
                zip(np.add(ACTV_KEYS[ACTV_KEYS >= insert_index], ins_shape[1]).tolist(),
                ACTV_VALUES[ACTV_KEYS >= insert_index].tolist())
            ))

            del ACTV_KEYS, ACTV_VALUES

            #   3) MERGE ALL DICTS TOGETHER USING LEFT_DICT | INS_OBJ | RIGHT_DICT
            LEFT_DICT[int(outer_idx)] = LEFT_DICT.pop(outer_idx) | INS_OBJ.pop(outer_idx) | RIGHT_DICT.pop(outer_idx)

        #   4) TERMINATE MODULE BY RETURNING DICT1
        return LEFT_DICT


def insert_inner(DICT1, index, LIST_OR_DICT_TO_INSERT, DICT_HEADER1=None, INSERT_HEADER=None,
                 header_axis=None, fxn=None):
    '''Insert an entry into all inner dictionaries at specified index with safeguards and header handling.'''

    # TEST CODE IS LOCATED IN sparse_dict_test_modules.insert_inner__function_test().

    # ASSUMES DICT1 IS "CLEAN"
    fxn = inspect.stack()[0][3] if fxn is None else fxn
    DICT1 = dict_init(DICT1, fxn)
    insufficient_dict_args_1(DICT1, fxn)
    dict1_outer_len = outer_len(DICT1)
    dict1_inner_len = inner_len(DICT1)

    DICT1 = core_insert_inner(DICT1, index, LIST_OR_DICT_TO_INSERT)

    del LIST_OR_DICT_TO_INSERT

    # IF INS_HEADER WAS NOT PASSED, MEASURE THE DELTA OF DICT1 BEFORE & AFTER INSERT TO FIND
    # OUT HOW LONG INS_OBJ WAS. USE THIS DELTA TO CREATE THE DUMMY HEADER.
    DICT_HEADER1, INSERT_HEADER = insert_outer_inner_header_handle(DICT_HEADER1, INSERT_HEADER, dict1_outer_len,
                           dict1_inner_len, header_axis, 1, inner_len_quick(DICT1) - dict1_inner_len, fxn)

    if not DICT_HEADER1 is None and not INSERT_HEADER is None:
        if header_axis == 1:
            # BECAUSE inner AND HEADER IS TIED TO INNER (AXIS 1), MODIFTY HEADER
            DICT_HEADER1 = np.hstack((DICT_HEADER1[...,:index], INSERT_HEADER, DICT_HEADER1[..., index:])).astype('<U200')
        # elif header_axis == 0: pass
            # NO CHANGE TO DICT_HEADER1, JUST RETURN THE ORIGINAL GIVEN OR WHAT CAME OUT OF header_handle

    if not DICT_HEADER1 is None: return DICT1, DICT_HEADER1
    else: return DICT1


def append_inner(DICT1, LIST_OR_DICT_TO_APPEND, DICT_HEADER1=None, INSERT_HEADER=None, header_axis=None, fxn=None):
    ''' Append an entry into all inner dictionaries in the last position.'''

    # TEST CODE IS LOCATED IN sparse_dict_test_modules.insert_inner__function_test().

    fxn = inspect.stack()[0][3] if fxn is None else fxn
    # VALIDATION IS HANDLED IN insert_inner

    if DICT1 == {}: index = 0
    else: index = inner_len(DICT1)

    if not (DICT_HEADER1 is None and INSERT_HEADER is None):
        DICT1, DICT_HEADER1 = insert_inner(DICT1, index, LIST_OR_DICT_TO_APPEND, DICT_HEADER1=DICT_HEADER1,
                                           INSERT_HEADER=INSERT_HEADER, header_axis=header_axis, fxn=fxn)
        return DICT1, DICT_HEADER1
    else:
        DICT1 = insert_inner(DICT1, index, LIST_OR_DICT_TO_APPEND, DICT_HEADER1=DICT_HEADER1, INSERT_HEADER=INSERT_HEADER,
                                            header_axis=header_axis, fxn=fxn)
        return DICT1


def split_outer(DICT1, index):
    '''Split before user-specified outer index; returns 2 sparse dictionaries as tuple'''
    fxn = inspect.stack()[0][3]
    DICT1 = dict_init(DICT1, fxn)
    insufficient_dict_args_1(DICT1, fxn)
    _sparse_dict_check(DICT1)

    _outer_len = outer_len(DICT1)

    if index >= _outer_len:
        DICT2 = {}
    elif index <= 0:
        DICT2 = deepcopy(DICT1)
        DICT1 = {}
    else:
        DICT2 = {int(_):DICT1[__] for _,__ in enumerate(range(index, _outer_len))}
        DICT1 = {int(_):DICT1[_] for _ in range(index)}

    return DICT1, DICT2


def split_inner(DICT1, index):
    '''Split before user-specified inner index; returns 2 sparse dictionaries as tuple'''
    fxn = inspect.stack()[0][3]
    DICT1 = dict_init(DICT1, fxn)
    insufficient_dict_args_1(DICT1, fxn)
    _sparse_dict_check(DICT1)

    _outer_len = outer_len(DICT1)
    _inner_len = inner_len(DICT1)

    if index >= _inner_len:
        DICT2 = {}
    elif index <= 0:
        DICT2 = deepcopy(DICT1)
        DICT1 = {}
    else:
        DICT2 = {}
        for outer_key in range(_outer_len):
            DICT2[int(outer_key)] = {}
            for new_key, inner_key in enumerate(range(index, _inner_len)):
                if inner_key in list(DICT1[outer_key].keys()):
                    DICT2[int(outer_key)][int(new_key)] = DICT1[outer_key][inner_key]
                # ENFORCE PLACEHOLDER RULES
                if inner_key == _inner_len - 1 and inner_key not in list(DICT1[outer_key].keys()):
                    DICT2[int(outer_key)][int(inner_key)] = 0

                if inner_key in list(DICT1[outer_key].keys()): del DICT1[outer_key][inner_key]

    return DICT1, DICT2


def multi_select_outer(DICT1, INDICES_AS_LIST):
    '''Build sparse dict from user-specified outer indices of given sparse dict'''
    fxn = inspect.stack()[0][3]
    DICT1 = dict_init(DICT1, fxn)
    insufficient_dict_args_1(DICT1, fxn)
    _sparse_dict_check(DICT1)

    NEW_DICT = {int(_):DICT1[__] for _,__ in enumerate(INDICES_AS_LIST)}

    return NEW_DICT


def core_multi_select_inner(DICT, INDICES_AS_LIST, as_inner=True, as_dict=True):
    '''Build sparse dict from user-specified inner indices of given sparse dict without safeguards.'''

    DICT1 = deepcopy(DICT)

    _outer_len = outer_len(DICT1)
    _inner_len = inner_len_quick(DICT1)

    if as_dict is True: NEW_OBJ = {}
    elif as_dict is False:
        if as_inner is True: NEW_OBJ = np.zeros((_outer_len, len(INDICES_AS_LIST)), dtype=np.float64)
        elif as_inner is False: NEW_OBJ = np.zeros((len(INDICES_AS_LIST), _outer_len), dtype=np.float64)

    if as_inner:
        for outer_key in range(_outer_len):
            if as_dict is True: NEW_OBJ[int(outer_key)] = {}
            for new_key, inner_key in enumerate(INDICES_AS_LIST):
                # ENFORCE PLACEHOLDER RULES
                if new_key == len(INDICES_AS_LIST) - 1:
                    NEW_OBJ[int(outer_key)][int(new_key)] = DICT1[outer_key].get(inner_key,0)
                elif inner_key in DICT1[outer_key]:
                    NEW_OBJ[int(outer_key)][int(new_key)] = DICT1[outer_key][inner_key]

    else:  # elif as_outer
        for new_key, old_inner_key in enumerate(INDICES_AS_LIST):
            if as_dict is True: NEW_OBJ[int(new_key)] = {}
            for old_outer_key in range(_outer_len):
                # ENFORCE PLACEHOLDER RULES
                if old_outer_key == _outer_len - 1:
                    NEW_OBJ[int(new_key)][int(old_outer_key)] = DICT1[old_outer_key].get(old_inner_key, 0)
                elif old_inner_key in DICT1[old_outer_key]:
                    NEW_OBJ[int(new_key)][int(old_outer_key)] = DICT1[old_outer_key][old_inner_key]

    return NEW_OBJ


def multi_select_inner(DICT1, INDICES_AS_LIST, as_inner=True, as_dict=True):
    '''Build sparse dict from user-specified inner indices of given sparse dict with safeguards.'''
    fxn = inspect.stack()[0][3]
    DICT1 = dict_init(DICT1, fxn)
    insufficient_dict_args_1(DICT1, fxn)
    _sparse_dict_check(DICT1)

    NEW_DICT = core_multi_select_inner(DICT1, INDICES_AS_LIST, as_inner=as_inner, as_dict=as_dict)

    return NEW_DICT


# END CREATION, HANDLING & MAINTENANCE ##################################################################################################
#########################################################################################################################################
#########################################################################################################################################


#########################################################################################################################################
#########################################################################################################################################
# ABOUT #################################################################################################################################


def display(DICT: dict, number_of_inner_dicts_to_print:int=float('inf')):
    """Print sparse dict to screen.


    """

    _insufficient_dict_args_1(DICT)
    DICT1 = _dict_init(DICT)

    _len = len(DICT1)

    while True:
        if _len > 50 and number_of_inner_dicts_to_print == float('inf'):
            __ = vui.validate_user_str(f'\nDICT has {_len} entries. Print all(p), pick number to print(n), abort(a) > ', 'NPA')
            if __ == 'A': break
            elif __ == 'N': num_rows = vui.validate_user_int(f'Enter number of inner dicts to print (of {_len}) > ', min=1, max=_len)
            elif __ == 'P': num_rows = _len
        else:
            num_rows = min(_len, number_of_inner_dicts_to_print)

        print()
        # IF DICT HASNT BEEN CLEANED, outer_keys MAY NOT BE IN SEQUENCE, SO ONLY PRINT VALID outer_keys
        VALID_OUTER_KEYS = set(DICT1.keys())
        print_count = 0
        outer_key = 0
        while print_count < num_rows:
            _ = outer_key
            if _ in VALID_OUTER_KEYS:
                print(f'{str(_)}:'.ljust(4) +f'{str(DICT1[_])[:100]}' + (f' ...' if len(str(DICT1[_])) > 70 else ''))
                print_count += 1
            outer_key += 1
        print()

        break


def core_find_constants(DICT1, orientation):
    '''Finds a column of constants. Returns dict/empty dict of non-zero constant indices, list/empty list of zero idxs.'''
    # RETURNS COLUMNS OF ZEROS FOR SUBSEQUENT HANDLING. len(COLUMNS OF CONSTANTS) SHOULD BE 1, BUT RETURN FOR HANDLING IF OTHERWISE.

    COLUMNS_OF_CONSTANTS, COLUMNS_OF_ZEROS = {}, []
    if orientation == 'COLUMN':
        for outer_idx in DICT1:
            _min, _max = min_max_({0: DICT1[outer_idx]})
            if _min != _max: continue
            else:  # _min == _max
                if _min == 0: COLUMNS_OF_ZEROS.append(outer_idx)
                else: COLUMNS_OF_CONSTANTS = COLUMNS_OF_CONSTANTS | {outer_idx: _min}

    elif orientation == 'ROW':
        for inner_idx in range(inner_len_quick(DICT1)):
            COL_HOLDER = np.fromiter((map(lambda x: DICT1[x].get(inner_idx, 0), DICT1)), dtype=np.float64)
            _min, _max = int(np.min(COL_HOLDER)), int(np.max(COL_HOLDER))

            if _min != _max: continue
            elif _min == _max and _min == 0: COLUMNS_OF_ZEROS.append(inner_idx)
            elif _min == _max: COLUMNS_OF_CONSTANTS = COLUMNS_OF_CONSTANTS | {inner_idx: _min}
        del COL_HOLDER

    return COLUMNS_OF_CONSTANTS, COLUMNS_OF_ZEROS


def find_constants(DICT1, orientation):
    '''Finds a column of constants with safeguards. Returns dict/empty dict of non-zero constant indices, list/empty list of zero idxs.'''
    fxn = inspect.stack()[0][3]
    DICT1 = dict_init(DICT1, fxn)
    insufficient_dict_args_1(DICT1, fxn)
    _sparse_dict_check(DICT1)

    orientation = akv.arg_kwarg_validater(orientation.upper(), 'outer_or_inner', ['ROW', 'COLUMN'],
                                             'sparse_dict', inspect.stack()[0][3])

    return core_find_contants(DICT1, orientation)




# END ABOUT #############################################################################################################################
#########################################################################################################################################
#########################################################################################################################################












