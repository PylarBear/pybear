import sys, inspect
import numpy as np
import sparse_dict as sd
from debug import get_module_name as gmn
from data_validation import arg_kwarg_validater as akv
from ML_PACKAGE._data_validation import list_dict_validater as ldv
from general_data_ops import find_constants as fc




def ML_find_constants(DATA_OBJECT, given_orientation):
    """Finds columns of constants for list-type arrays and sparse dictionaries."""

    this_module = gmn.get_module_name(str(sys.modules[__name__]))
    fxn = inspect.stack()[0][3]

    given_orientation = akv.arg_kwarg_validater(given_orientation, 'given_orientation', ['ROW','COLUMN'], this_module, fxn)
    data_format, DATA_OBJECT = ldv.list_dict_validater(DATA_OBJECT, 'DATA_OBJECT')

    if data_format=='ARRAY':
        COLUMNS_OF_CONSTANTS, COLUMNS_OF_ZEROS = fc.find_constants(DATA_OBJECT, given_orientation)
    elif data_format=='SPARSE_DICT':
        COLUMNS_OF_CONSTANTS, COLUMNS_OF_ZEROS = sd.core_find_constants(DATA_OBJECT, given_orientation)

    del this_module, fxn, data_format

    return COLUMNS_OF_CONSTANTS, COLUMNS_OF_ZEROS








if __name__ == '__main__':

    # MODULE & TEST CODE VERIFIED GOOD 7/1/23

    import time
    from copy import deepcopy

    _rows, _cols = 100, 50
    _orient = 'COLUMN'

    BASE_DATA = np.random.randint(1, 10, (_rows if _orient == 'ROW' else _cols, _cols if _orient == 'ROW' else _rows))

    ctr = 0
    for data_format in ['ARRAY', 'SPARSE_DICT']:
        for data_orientation in ['ROW','COLUMN']:
            for COLUMNS_OF_ZEROS in ([], [0,_cols//2,_cols-1]):
                for COLUMNS_OF_CONSTANTS in ({}, {1:1, _cols//2+1:2, _cols-2:3}):
                    ctr += 1
                    print(f'Running trial {ctr} of {2*2*2*2}...')



                    GIVEN_DATA = BASE_DATA.copy()
                    if _orient=='COLUMN':
                        GIVEN_DATA[COLUMNS_OF_ZEROS, :] = 0
                        for k,v in COLUMNS_OF_CONSTANTS.items():
                            GIVEN_DATA[k, :] = v
                    elif _orient=='ROW':
                        GIVEN_DATA = GIVEN_DATA[:, COLUMNS_OF_ZEROS] = 0
                        GIVEN_DATA[:, list(COLUMNS_OF_CONSTANTS.keys())] = list(COLUMNS_OF_CONSTANTS.values())

                    if data_orientation != _orient: GIVEN_DATA = GIVEN_DATA.transpose()

                    if data_format=='SPARSE_DICT': GIVEN_DATA = sd.zip_list_as_py_int(GIVEN_DATA)

                    ACT_COLUMNS_OF_CONSTANTS, ACT_COLUMNS_OF_ZEROS = ML_find_constants(GIVEN_DATA, data_orientation)

                    # ANSWER KEY ##################################################################################################

                    EXP_COLUMNS_OF_CONSTANTS = deepcopy(COLUMNS_OF_CONSTANTS)
                    EXP_COLUMNS_OF_ZEROS = COLUMNS_OF_ZEROS.copy()

                    # END ANSWER KEY ##################################################################################################

                    if not np.array_equiv(EXP_COLUMNS_OF_ZEROS, ACT_COLUMNS_OF_ZEROS):
                        print(f'\033[91m')
                        print(f'ACT COLUMNS_OF_ZEROS = \n')
                        print(ACT_COLUMNS_OF_ZEROS)
                        print(f'\nEXP COLUMNS_OF_ZEROS = \n')
                        print(EXP_COLUMNS_OF_ZEROS)
                        time.sleep(0.5)
                        raise Exception(f'*** EXP_COLUMNS_OF_ZEROS, ACT_COLUMNS_OF_ZEROS NOT EQUAL ***')

                    # FUDGE THESE TO BE SPARSE DICTS EVEN THO THEY ARENT REALLY
                    if not sd.core_sparse_equiv({0: EXP_COLUMNS_OF_CONSTANTS}, {0: ACT_COLUMNS_OF_CONSTANTS}):
                        print(f'\033[91m')
                        print(f'ACT COLUMNS_OF_CONSTANTS = \n')
                        print(ACT_COLUMNS_OF_CONSTANTS)
                        print(f'\nEXP COLUMNS_OF_CONSTANTS = \n')
                        print(EXP_COLUMNS_OF_CONSTANTS)
                        time.sleep(0.5)
                        raise Exception(f'*** EXP_COLUMNS_OF_CONSTANTS, ACT_COLUMNS_OF_CONSTANTS NOT EQUAL ***')


    print(f'\033[92m\n*** ALL TESTS PASSED ***\033[0m')
    from general_sound import winlinsound as wls
    for _ in range(3): wls.winlinsound(888, 1); time.sleep(0.5)



















































