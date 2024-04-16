import sys
from debug import get_module_name as gmn
from data_validation import validate_user_input as vui
from ML_PACKAGE._data_validation import list_dict_validater as ldv
from general_data_ops import manage_intercept as mint
from MLObjects.SupportObjects import master_support_object_dict as msod




class InvalidShapeError(Exception): pass


def ML_manage_intercept(DATA_OBJECT, given_orientation, DATA_FULL_SUPOBJ_OR_HEADER=None):
    """Finds columns of constants, including zeros, for list-type arrays and sparse dictionaries. Returns a list of
    column indices user chose to delete in desc order, dictionary of constant columns still in, and list of columns of
    zeros still in."""

    # VALIDATE HEADER OF FULL SUP OBJ AND CONVERT TO HEADER #################################################################
    DATA_FULL_SUPOBJ_OR_HEADER = ldv.list_dict_validater(DATA_FULL_SUPOBJ_OR_HEADER, 'DATA_FULL_SUPOBJ_OR_HEADER')[1]


    if DATA_FULL_SUPOBJ_OR_HEADER is None: ACTV_HDR = None
    else:
        if len(DATA_FULL_SUPOBJ_OR_HEADER) == 1: ACTV_HDR = DATA_FULL_SUPOBJ_OR_HEADER[0]
        elif len(DATA_FULL_SUPOBJ_OR_HEADER) == len(msod.master_support_object_dict()):
            ACTV_HDR = DATA_FULL_SUPOBJ_OR_HEADER[msod.QUICK_POSN_DICT()["HEADER"]]
        else: raise InvalidShapeError(f'DATA_FULL_SUPOBJ_OR_HEADER ROWS ({len(DATA_FULL_SUPOBJ_OR_HEADER)}) DO NOT MATCH '
                                    f'HEADER ONLY OR FULL SUPPORT OBJECT')
    # END VALIDATE HEADER OF FULL SUP OBJ AND CONVERT TO HEADER #############################################################


    return mint.manage_intercept(DATA_OBJECT, given_orientation, ACTV_HDR)
    
    
    
    
    








if __name__ == '__main__':

    import numpy as np
    from general_list_ops import manual_num_seq_list_fill as mnslf
    from MLObjects.TestObjectCreators import test_header as th

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
        ZEROS = list(map(int, ZEROS))

        num_cons_cols = vui.validate_user_int(f'\nEnter number of constant columns > ', min=0, max=_cols - len(ZEROS))
        CONSTANTS = {}

        if num_cons_cols > 0:
            ctr = 0
            while len(CONSTANTS) < num_cons_cols:
                ctr += 1
                col_idx = vui.validate_user_int(f'\nEnter column index ({ctr} of {num_cons_cols}) > ', min=0,
                                                max=_cols - 1)
                if col_idx in ZEROS + list(CONSTANTS.keys()):
                    print(f'\n*** INDEX IS ALREADY OCCUPIED BY COLUMN OF ZEROS OR CONSTANTS, ENTER A DIFFERENT COLUMN INDEX ***\n')
                    ctr -= 1; continue
                else: value = vui.validate_user_float(f'Enber value > ')

                CONSTANTS = CONSTANTS | {col_idx: value}

            del ctr, col_idx, value

        GIVEN_DATA = BASE_DATA.copy()
        if _orient == 'COLUMN':
            GIVEN_DATA[ZEROS, :] = 0
            for k, v in CONSTANTS.items(): GIVEN_DATA[k, :] = v
        elif _orient == 'ROW':
            GIVEN_DATA[:, ZEROS] = 0
            for k, v in CONSTANTS.items(): GIVEN_DATA[:, k] = v

        del num_zero_cols, num_cons_cols

        if vui.validate_user_str(f'\nPass FULL_SUPOBJ? (y/n) > ', 'YN') == 'Y':
            FULL_SUPOBJ = msod.build_random_support_object(_cols)
            FULL_SUPOBJ[msod.QUICK_POSN_DICT()["HEADER"], :] = th.test_header(_cols)[0]
            FULL_SUPOBJ[[msod.QUICK_POSN_DICT()["VALIDATEDDATATYPES"], msod.QUICK_POSN_DICT()["MODIFIEDDATATYPES"]], :] = 'INT'
        else:
            FULL_SUPOBJ = None

        # END RIG BASE_DATA TO USER SELECTED COLUMNS OF CONSTANTS AND COLUMNS OF ZEROS ######################################

        print(f'\n*** DATA SETUP COMPLETE, ENTERING {gmn.get_module_name(str(sys.modules[__name__]))} TEST ***\n')

        TO_DELETE, COLUMNS_OF_CONSTANTS_AS_DICT, COLUMNS_OF_ZEROS_AS_LIST = ML_manage_intercept(GIVEN_DATA, _orient, FULL_SUPOBJ)

        print(f'\nSTARTING ACTUAL COLUMNS_OF_ZEROS = {ZEROS}')
        print(f'STARTING ACTUAL COLUMNS_OF_CONSTANTS = {CONSTANTS}')
        print(f'TO_DELETE = {TO_DELETE}')
        print(f'COLUMNS_OF_CONSTANTS_AS_DICT = {COLUMNS_OF_CONSTANTS_AS_DICT}')
        print(f'COLUMNS_OF_ZEROS_AS_LIST = {COLUMNS_OF_ZEROS_AS_LIST}')

        if vui.validate_user_str(f'\nRun another test(t) or quit(q) > ', 'TQ') == 'Q': break




















