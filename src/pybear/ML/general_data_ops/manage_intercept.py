import sys, inspect
import numpy as np
import sparse_dict as sd
from debug import get_module_name as gmn
from data_validation import arg_kwarg_validater as akv, validate_user_input as vui
from ML_PACKAGE._data_validation import list_dict_validater as ldv
from general_list_ops import list_select as ls
from general_data_ops import find_constants as fc, get_shape as gs


class InvalidShapeError(Exception): pass


def manage_intercept(DATA_OBJECT, given_orientation, HEADER=None):
    """Finds columns of constants, including zeros, for list-type arrays and sparse dictionaries. Returns a list of
    column indices user chose to delete in desc order, dictionary of constant columns still in, and list of columns of
    zeros still in."""

    this_module = gmn.get_module_name(str(sys.modules[__name__]))
    fxn = inspect.stack()[0][3]

    DATA_OBJECT = ldv.list_dict_validater(DATA_OBJECT, 'DATA OBJECT')[1]

    given_orientation = akv.arg_kwarg_validater(given_orientation, 'given_orientation', ['ROW', 'COLUMN'],
                                                this_module, fxn)

    data_cols = gs.get_shape('DATA_OBJECT', DATA_OBJECT, given_orientation)[1]

    HEADER = ldv.list_dict_validater(HEADER, 'HEADER')[1]


    if HEADER is None: ACTV_HDR = [f'COLUMN_{_}' for _ in range(1, data_cols+1)]
    else:
        if data_cols != HEADER.shape[1]:
            raise InvalidShapeError(f'DATA COLUMNS ({data_cols}) DOES NOT EQUAL HEADER COLUMNS ({HEADER.shape[1]})')
        if HEADER.shape[0] == 1: ACTV_HDR = HEADER[0]
        else: raise InvalidShapeError(f'INVALID SHAPE {HEADER.shape} FOR HEADER')

    del data_cols

    # FIND COLUMNS OF CONSTANTS
    if isinstance(DATA_OBJECT, (list, tuple, np.ndarray)):
        COLUMNS_OF_CONSTANTS_AS_DICT, COLUMNS_OF_ZEROS_AS_LIST = fc.find_constants(DATA_OBJECT, given_orientation)
    elif isinstance(DATA_OBJECT, dict):
        COLUMNS_OF_CONSTANTS_AS_DICT, COLUMNS_OF_ZEROS_AS_LIST = sd.core_find_constants(DATA_OBJECT, given_orientation)


    # MANAGE ANOMALOUS INTERCEPTS IN DATA ##############################################################################
    while True:
        TO_DELETE = {}  # BUILD A DICTIONARY (LIKE COLUMNS OF CONSTANTS) WITH col_idx AS key AND VALUES AS value
        if len(COLUMNS_OF_ZEROS_AS_LIST) > 0:
            print(f'\n *** DATA HAS COLUMN(S) OF ZEROS. ***\n')
            [print(f'Column index {_}, header = {ACTV_HDR[_]}') for _ in COLUMNS_OF_ZEROS_AS_LIST]
            if vui.validate_user_str(f'\nDelete all(d) or ignore(i) > ', 'DI') == 'D':
                TO_DELETE = TO_DELETE | {k: 0 for k in COLUMNS_OF_ZEROS_AS_LIST}

        if len(COLUMNS_OF_CONSTANTS_AS_DICT) > 0:
            print(f'\n *** DATA HAS COLUMN(S) OF CONSTANTS. *** \n')
            LIST_OF_CONSTANTS = list(f'Column index {k}, header = {ACTV_HDR[k]}, value = {v}' for k, v in
                                     COLUMNS_OF_CONSTANTS_AS_DICT.items())
            [print(_) for _ in LIST_OF_CONSTANTS]
            _ = vui.validate_user_str(f'\nSelect indices to delete(s), ignore(i) > ', 'SI')
            if _ == 'S':
                __ = ls.list_multi_select(LIST_OF_CONSTANTS, f'\nSelect columns to delete', 'idx')
                TO_DELETE = TO_DELETE | {k: v for k, v in COLUMNS_OF_CONSTANTS_AS_DICT.items() if k in \
                                         np.fromiter(COLUMNS_OF_CONSTANTS_AS_DICT.keys(), dtype=np.int32)[__]}
                del __

            del LIST_OF_CONSTANTS

        if len(TO_DELETE) > 0:
            for _ in range(max(TO_DELETE.keys()) + 1):  # SORT TO_DELETE
                if _ in TO_DELETE: TO_DELETE[_] = TO_DELETE.pop(_)

        # REMOVE SELECTED IDXS IN TO_DELETE FROM COLUMNS_OF_CONSTANTS_AS_DICT AND COLUMNS_OF_ZEROS_AS_LIST TO SEE IF ANYTHING LEFT
        if len(TO_DELETE) > 0:
            COLUMNS_OF_CONSTANTS_AS_DICT = {k: v for k, v in COLUMNS_OF_CONSTANTS_AS_DICT.items() if k not in TO_DELETE}
            COLUMNS_OF_ZEROS_AS_LIST = [_ for _ in COLUMNS_OF_ZEROS_AS_LIST if _ not in TO_DELETE.keys()]

        # DISPLAY USER SELECTIONS ##########################################################################################
        if len(COLUMNS_OF_ZEROS_AS_LIST) > 0:
            print(f'\n *** WARNING: DATA WILL STILL HAVE {len(COLUMNS_OF_ZEROS_AS_LIST)} COLUMN(S) OF ZEROS *** \n')

        if len(COLUMNS_OF_CONSTANTS_AS_DICT) > 1:
            print(f'\n *** WARNING: DATA WILL STILL HAVE {len(COLUMNS_OF_CONSTANTS_AS_DICT)} COLUMN(S) OF CONSTANTS *** \n')

        print(f'\nUser opted to delete:')
        [print(f'Column index {k}, header = {ACTV_HDR[k]}, value = {v}') for k, v in TO_DELETE.items()]
        print(f'\nUser opted to keep:')
        if len(COLUMNS_OF_ZEROS_AS_LIST) + len(COLUMNS_OF_CONSTANTS_AS_DICT) == 0:
            print('None')
        else:
            DUM_DICT = {_: 0 for _ in COLUMNS_OF_ZEROS_AS_LIST} | COLUMNS_OF_CONSTANTS_AS_DICT
            for _ in range(max(DUM_DICT.keys()) + 1):  # SORT DUM_DICT
                if _ in DUM_DICT: DUM_DICT[_] = DUM_DICT.pop(_)
            [print(f'Column index {k}, header = {ACTV_HDR[k]}, value = {v}') for k, v in DUM_DICT.items()]
            del DUM_DICT
        # END DISPLAY USER SELECTIONS #######################################################################################

        if vui.validate_user_str(f'\nAccept? (y/n) (no restarts selection process) > ', 'YN') == 'Y': break

    TO_DELETE = np.fromiter(TO_DELETE.keys(), dtype=np.int32)
    TO_DELETE.sort()
    TO_DELETE = np.flip(TO_DELETE)  # PROBABLY GOING TO INTERATE THRU COL IDXS BACKWARDS

    del ACTV_HDR

    for deleted_idx in TO_DELETE:
        COLUMNS_OF_CONSTANTS_AS_DICT = {k-1 if k > deleted_idx else k:v for k,v in COLUMNS_OF_CONSTANTS_AS_DICT.items()}
        COLUMNS_OF_ZEROS_AS_LIST = [idx-1 if idx > deleted_idx else idx for idx in COLUMNS_OF_ZEROS_AS_LIST]

    # TO_DELETE = LIST OF INDICES TO BE DELETED
    # COLUMNS_OF_CONSTANTS_AS_DICT = DICT OF REMAINING COLUMNS OF CONSTANTS AFTER DELETE IS APPLIED
    # COLUMNS_OF_ZEROS_AS_LIST = LIST OF REMAINING COLUMNS OF CONSTANTS AFTER DELETE IS APPLIED
    return TO_DELETE, COLUMNS_OF_CONSTANTS_AS_DICT, COLUMNS_OF_ZEROS_AS_LIST

    #   END MANAGE ANOMALOUS INTERCEPTS IN DATA ########################################################################





if __name__ == '__main__':

    from MLObjects.TestObjectCreators import test_header as th
    from general_list_ops import manual_num_seq_list_fill as mnslf

    # THIS CODE IS ALWAYS THE SAME NO MATTER WHAT DATA FORMAT / ORIENTATION IS, BECAUSE EVERYTHING THAT THIS MODULE
    # WORKS ON IS WHAT COMES OUT OF ML_find_constants(), AND IT ALWAYS RETURNS COLUMNS_OF_CONSTANTS_AS_DICT,
    # COLUMNS_OF_ZEROS_AS_LIST AS dict/list. SO ONLY NEED TO TEST WITH ONE FORMAT/ORIENTATION.

    # TO TEST HANDLING OF DATA FORMATS / ORIENTATIONS, RUN TEST IN ML_find_constants

    _rows, _cols = 10, 5
    _orient = 'COLUMN'

    BASE_DATA = np.random.randint(1, 10, (_rows if _orient == 'ROW' else _cols, _cols if _orient == 'ROW' else _rows))

    while True:

        # RIG BASE_DATA TO USER SELECTED COLUMNS OF CONSTANTS AND COLUMNS OF ZEROS #########################################
        num_zero_cols = vui.validate_user_int(f'\nEnter number of zero columns > ', min=0, max=_cols)
        if num_zero_cols == 0: ZEROS = []
        else: ZEROS = mnslf.manual_num_seq_list_fill('zero column index', [], num_zero_cols, min=0, max=_cols - 1)
        ZEROS = sorted(list(map(int, ZEROS)))

        num_cons_cols = vui.validate_user_int(f'\nEnter number of constant columns > ', min=0, max=_cols - len(ZEROS))
        CONSTANTS = {}

        if num_cons_cols > 0:
            ctr = 0
            while len(CONSTANTS) < num_cons_cols:
                ctr += 1
                col_idx = vui.validate_user_int(f'\nEnter column index ({ctr} of {num_cons_cols}) > ', min=0, max=_cols - 1)
                if col_idx in ZEROS + list(CONSTANTS.keys()):
                    print(f'\n*** INDEX IS ALREADY OCCUPIED BY COLUMN OF ZEROS OR CONSTANT, ENTER A DIFFERENT COLUMN INDEX ***\n')
                    ctr -= 1; continue
                else: value = vui.validate_user_float(f'Enber value > ')

                CONSTANTS = CONSTANTS | {col_idx: value}

            del ctr, col_idx, value

        for key in range(max(CONSTANTS)+1):
            if key in CONSTANTS: CONSTANTS[key] = CONSTANTS.pop(key)


        GIVEN_DATA = BASE_DATA.copy()
        if _orient == 'COLUMN':
            GIVEN_DATA[ZEROS, :] = 0
            for k, v in CONSTANTS.items(): GIVEN_DATA[k, :] = v
        elif _orient == 'ROW':
            GIVEN_DATA[:, ZEROS] = 0
            for k, v in CONSTANTS.items(): GIVEN_DATA[:, k] = v

        del num_zero_cols, num_cons_cols

        if vui.validate_user_str(f'\nPass header (y/n) > ', 'YN') == 'Y': HEADER = th.test_header(_cols)
        else: HEADER = None

        # END RIG BASE_DATA TO USER SELECTED COLUMNS OF CONSTANTS AND COLUMNS OF ZEROS ######################################


        print(f'\n*** DATA SETUP COMPLETE, ENTERING {gmn.get_module_name(str(sys.modules[__name__]))} TEST ***\n')

        TO_DELETE, COLUMNS_OF_CONSTANTS_AS_DICT, COLUMNS_OF_ZEROS_AS_LIST = manage_intercept(GIVEN_DATA, _orient, HEADER)

        print(f'\nSTARTING ACTUAL COLUMNS_OF_ZEROS = {ZEROS}')
        print(f'STARTING ACTUAL COLUMNS_OF_CONSTANTS = {CONSTANTS}')
        print(f'TO_DELETE = {TO_DELETE}')
        print(f'COLUMNS_OF_CONSTANTS_AS_DICT = {COLUMNS_OF_CONSTANTS_AS_DICT}')
        print(f'COLUMNS_OF_ZEROS_AS_LIST = {COLUMNS_OF_ZEROS_AS_LIST}')
        print(f'\033[92mINDICES IN CONSTANTS_AS_DICT & ZEROS_AS_LIST SHOULD BE SHIFTED DOWN BY NUMBER OF COLUMNS TO BE DELETED TO THE LEFT.\033[0m')

        if vui.validate_user_str(f'\nRun another test(t) or quit(q) > ', 'TQ') == 'Q': break



















