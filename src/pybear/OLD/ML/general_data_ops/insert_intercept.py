import sys, inspect
import numpy as np
import sparse_dict as sd
from ML_PACKAGE._data_validation.list_dict_validater import list_dict_validater, NotArrayOrSparseDictError
from data_validation.arg_kwarg_validater import arg_kwarg_validater
from debug import get_module_name as gmn
from general_data_ops import get_shape as gs


class DisallowedError(Exception): pass

class ColIdxOutOfRangeError(Exception): pass

class NonNumericalError(Exception): pass

class IncongruencyError(Exception): pass


def insert_intercept(DATA, data_given_orientation, col_idx, value_to_insert=None, HEADER=None, header_string=None):
    """Append intercept to data and optionally header"""

    this_module = gmn.get_module_name(str(sys.modules[__name__]))
    fxn = inspect.stack()[0][3]

    DATA = list_dict_validater(DATA, "DATA")[1]

    data_given_orientation = arg_kwarg_validater(data_given_orientation, 'data_given_orientation',
                                                     ['ROW', 'COLUMN'], this_module, fxn)

    data_rows, data_cols = gs.get_shape('DATA', DATA, data_given_orientation)

    if not col_idx in range(data_cols+1):
        raise ColIdxOutOfRangeError(f'GIVEN col_idx ({col_idx}) IS OUT OF RANGE, MUST BE IN RANGE 0, {data_cols}', fxn)

    if not value_to_insert is None:
        if not True in map(lambda x: x in str(type(value_to_insert)).upper(), ['INT', 'FLOAT']):
            raise NonNumericalError(f'value_to_insert MUST BE int OR float')
    else: value_to_insert = 1

    if value_to_insert == 0:
        raise DisallowedError(f'value_to_insert CANNOT BE ZERO.')

    HEADER = list_dict_validater(HEADER, "HEADER")[1]
    HEADER = HEADER.reshape((1,-1)) if not HEADER is None else None

    if not HEADER is None:
        if HEADER.shape[1] != data_cols: raise IncongruencyError(f'DATA AND HEADER DO NOT HAVE EQUAL COLUMNS')
        if HEADER.shape[0] != 1: raise IncongruencyError(f'HEADER HAS ILLEGAL SHAPE {HEADER.shape}')


    if not header_string is None and not isinstance(header_string, str):
        raise DisallowedError(f'IF header_string IS PASSED, IT MUST BE str')


    if isinstance(DATA, np.ndarray):
        DATA = np.insert(DATA, col_idx, value_to_insert, axis=0 if data_given_orientation=='COLUMN' else 1)
    elif isinstance(DATA, dict):
        if data_given_orientation=='COLUMN':
            DATA = sd.insert_outer(DATA, col_idx, np.full(data_rows, value_to_insert))
        elif data_given_orientation=='ROW':
            DATA = sd.insert_inner(DATA, col_idx, np.full(data_rows, value_to_insert))

    if not HEADER is None:
        if not header_string is None: HEADER = np.insert(HEADER, col_idx, header_string, axis=1)
        elif header_string is None: HEADER = np.insert(HEADER, col_idx, 'INTERCEPT', axis=1)

    if HEADER is None: return DATA
    else: return DATA, HEADER



















if __name__ == '__main__':

    # MODULE AND TEST CODE VERIFIED GOOD 6/28/23

    import time
    from copy import deepcopy
    from MLObjects.TestObjectCreators import test_header as th
    from general_sound import winlinsound as wls

    print_green = lambda words: print(f'\033[92m{words}\033[0m')
    print_red = lambda words: print(f'\033[91m{words}\033[0m')

    calling_module = gmn.get_module_name(str(sys.modules[__name__]))

    ##########################################################################################################################
    # TEST DATA VALIDATION EXCEPTIONS ########################################################################################

    test_name = 'data _validation exceptions'
    print(f'Running {test_name}...')

    _rows, _cols = 100, 80
    given_orientation = 'COLUMN'

    DATA = np.random.randint(1, 10, (
    _cols if given_orientation == 'COLUMN' else _rows, _rows if given_orientation == 'COLUMN' else _cols))

    fails = 0

    BASE_KWARGS = {
                    'DATA': DATA,
                    'data_given_orientation': given_orientation,
                    'col_idx': _cols,
                    'value_to_insert': None,
                    'HEADER': None,
                    'header_string': None,
    }

    # DATA #####################################################################################################################

    # class IllegalObjectDatatypeError(Exception): pass

    print_green(f'\nTesting DATA other than list-type or sparse dict excepts...')
    KWARGS = deepcopy(BASE_KWARGS); KWARGS['DATA'] = 'DUM'
    try: insert_intercept(**KWARGS); fails+=1; print_red(f'Did not except for DATA other than list-type or sparse dict.')
    except NotArrayOrSparseDictError: print_green(f'Pass.')
    except: fails +=1; print_red(f'\nTesting DATA other than list-type or sparse dict excepts FAILED FOR {sys.exc_info()[0]}, not NotArrayOrSparseDictError')

    # END DATA #####################################################################################################################

    # data_given_orientation ###########################################################################################

    print_green(f'\nTesting data_given_orientation other than ROW or COLUMN excepts...')
    KWARGS = deepcopy(BASE_KWARGS); KWARGS['data_given_orientation'] = 'DUM'
    try: insert_intercept(**KWARGS); fails += 1; print_red(f'Fail. data_given_orientation other than ROW or COLUMN did not except.')
    except DisallowedError: print_green(f'Pass.')
    except: fails+=1; print_red(f'Testing data_given_orientation other than ROW or COLUMN excepted for {sys.exc_info()[0]}, not DisallowedError')

    # END data_given_orientation #######################################################################################

    # col_idx       # ###################################################################################################
    print_green(f'\nTesting negative col_idx excepts...')
    KWARGS = deepcopy(BASE_KWARGS); KWARGS['col_idx'] = -1
    try: insert_intercept(**KWARGS); fails+=1; print_red(f'Accepted a negative col_idx.')
    except ColIdxOutOfRangeError: print_green(f'Pass.')
    except: fails+=1; print_red(f'Testing negative col_idx excepted for {sys.exc_info()[0]}, not ColIdxOutOfRangeError.')


    print_green(f'\nTesting col_idx > cols excepts...')
    KWARGS = deepcopy(BASE_KWARGS); KWARGS['col_idx'] = _cols + 1
    try: insert_intercept(**KWARGS); fails+=1; print_red(f'Accepted col_idx > cols.')
    except ColIdxOutOfRangeError: print_green(f'Pass.')
    except: fails+=1; print_red(f'Testing col_idx > cols excepted for {sys.exc_info()[0]}, not ColIdxOutOfRangeError.')
    # END col_idx     # ########################################################################################################


    # value_to_insert  # ####################################################################################################
    print_green(f'\nTesting value_to_insert other than number or None excepts...')
    KWARGS = deepcopy(BASE_KWARGS); KWARGS['value_to_insert'] = 'DUM'
    try: insert_intercept(**KWARGS); fails += 1; print_red(f'Fail. value_to_insert other than number or list did not except.')
    except NonNumericalError: print_green(f'Pass.')
    except: fails+=1; print_red(f'Testing value_to_insert other than number or None excepted for {sys.exc_info()[0]}, not NonNumericalError')


    print_green(f'\nTesting value_to_insert==0 excepts...')
    KWARGS = deepcopy(BASE_KWARGS); KWARGS['value_to_insert'] = 0
    try: insert_intercept(**KWARGS); fails += 1; print_red(f'Fail. value_to_insert==0 did not except.')
    except DisallowedError: print_green(f'Pass.')
    except: fails+=1; print_red(f'Testing value_to_insert==0 excepted for {sys.exc_info()[0]}, not DisallowedError')

    # END value_to_insert  # #################################################################################################


    # HEADER & header_string ############################################################################################

    # SET BASE_KWARGS value_to_insert PERMANENTLY TO TEST HEADER THINGS
    BASE_KWARGS['value_to_insert'] = 2

    print_green(f'\nTesting HEADER excepts if not a list-type...')
    KWARGS = deepcopy(BASE_KWARGS)
    KWARGS['HEADER'] = 'DUM'
    try: insert_intercept(**KWARGS); fails += 1; print_red(f'Fail. HEADER did not except when passed as non-list-type.')
    except NotArrayOrSparseDictError: print_green(f'Pass.')
    except: fails+=1; print_red(f'Testing HEADER excepts if not a list-type excepted for {sys.exc_info()[0]}, not NotArrayOrSparseDictError')


    # TESTING FOR HEADER
    print_green(f'\nTesting HEADER excepts if not sized as header (1 row)...')
    KWARGS = deepcopy(BASE_KWARGS)
    KWARGS['HEADER'] = np.tile(th.test_header(_cols), (3, 1))
    try:
        insert_intercept(**KWARGS); fails += 1
        print_red(f'Fail. HEADER did not except for incorrect rows for a header (1).')
    except IncongruencyError: print_green(f'Pass.')
    except: fails+=1; print_red(f'Fail. HEADER incorrect rows for a header (1). excepted for {sys.exc_info()[0]}, not IncongruencyError')


    print_green(f'\nTesting HEADER excepts if not enough columns...')
    KWARGS = deepcopy(BASE_KWARGS)
    KWARGS['HEADER'] = th.test_header(_cols)[..., :-1]
    try: insert_intercept(**KWARGS); fails += 1; print_red(f'Fail. HEADER did not except when not enough columns.')
    except IncongruencyError: print_green(f'Pass.')
    except: fails+=1; print_red(f'Testing HEADER not enough columns excepted for {sys.exc_info()[0]}, not IncongruencyError')


    print_green(f'\nTesting HEADER excepts if too many columns...')
    KWARGS = deepcopy(BASE_KWARGS)
    KWARGS['HEADER'] = np.insert(th.test_header(_cols), 0, 'EXTRA COLUMN', axis=1)
    try: insert_intercept(**KWARGS); fails += 1; print_red(f'Fail. HEADER did not except when too many columns.')
    except IncongruencyError: print_green(f'Pass.')
    except: print_red(f'Testing HEADER too many columns excepted for {sys.exc_info()[0]}, not IncongruencyError.')


    # SET BASE_KWARGS HEADER TEMPORARILY AS A HEADER TO TEST HEADER THINGS
    BASE_KWARGS["HEADER"] = th.test_header(_cols)


    print_green(f'\nTesting header_string excepts if not a str...')
    KWARGS = deepcopy(BASE_KWARGS)
    KWARGS["header_string"] = 8
    try: insert_intercept(**KWARGS); fails += 1; print_red(f'Fail. header_string did not except when passed a non-string.')
    except DisallowedError: print_green(f'Pass.')
    except: fails+=1; print_red(f'Testing header_string if not a str excepted or {sys.exc_info()[0]}, not DisallowedError.')

    # END HEADER & header_string ##################################################################################################

    # END TEST DATA VALIDATION EXCEPTIONS ########################################################################################
    ##########################################################################################################################




    # TEST FOR ACCURACY
    print(f'\nStart tests for accuracy...')

    _rows, _cols = 60, 40

    MASTER_DATA_GIVEN_ORIENTATION = ['ROW', 'COLUMN']
    MASTER_DATA_GIVEN_FORMAT = ['ARRAY', 'SPARSE_DICT']
    MASTER_COL_IDX = [0, _cols // 2, _cols]
    MASTER_VALUE_TO_INSERT = [1, -1]
    MASTER_HEADER_IS_GIVEN = [True, False]
    MASTER_HEADER_STR_IS_GIVEN = [True, False]


    total_trials = np.product(list(map(len, (MASTER_DATA_GIVEN_FORMAT, MASTER_DATA_GIVEN_ORIENTATION, MASTER_COL_IDX,
                                             MASTER_VALUE_TO_INSERT, MASTER_HEADER_IS_GIVEN, MASTER_HEADER_STR_IS_GIVEN))))


    PRIMAL_DATA = np.random.randint(0, 10, (_rows, _cols))
    PRIMAL_HEADER = th.test_header(_cols)

    ctr = 0
    for data_given_orientation in MASTER_DATA_GIVEN_ORIENTATION:
        if data_given_orientation == 'COLUMN':
            BASE_DATA = PRIMAL_DATA.copy().transpose()
        else:
            BASE_DATA = PRIMAL_DATA.copy()
        for data_given_format in MASTER_DATA_GIVEN_FORMAT:
            if data_given_format == 'SPARSE_DICT': BASE_DATA = sd.zip_list_as_py_int(BASE_DATA)
            for col_idx in MASTER_COL_IDX:
                for given_value_to_insert in MASTER_VALUE_TO_INSERT:
                    for header_is_given in MASTER_HEADER_IS_GIVEN:
                        for header_string_is_given in MASTER_HEADER_STR_IS_GIVEN:
                            ctr += 1
                            print(f'\nRunning trial {ctr} of {total_trials}...')

                            GIVEN_DATA = deepcopy(BASE_DATA) if isinstance(BASE_DATA,
                                                                           dict) else BASE_DATA.copy()

                            # HEADER
                            if header_is_given: GIVEN_HEADER = PRIMAL_HEADER.copy()
                            elif not header_is_given: GIVEN_HEADER = None

                            # header_string
                            if header_string_is_given: given_header_string = 'GIVEN INTERCEPT'
                            elif not header_string_is_given: given_header_string = None


                            # RUN FUNCTION ###############################################################################
                            if header_is_given:
                                ACT_DATA, ACT_HEADER = insert_intercept(GIVEN_DATA, data_given_orientation, col_idx,
                                value_to_insert=given_value_to_insert, HEADER=GIVEN_HEADER, header_string=given_header_string
                                )

                            elif not header_is_given:
                                ACT_DATA = insert_intercept(GIVEN_DATA, data_given_orientation, col_idx,
                                value_to_insert=given_value_to_insert, HEADER=GIVEN_HEADER, header_string=given_header_string
                                )
                                ACT_HEADER = None

                            if data_given_format == 'SPARSE_DICT': ACT_DATA = sd.unzip_to_ndarray_float64(ACT_DATA)[0].astype(np.int32)

                            # END RUN FUNCTION ############################################################################



                            # ANSWER KEY #######################################################################################################

                            ANSWER_DATA = PRIMAL_DATA.copy()
                            if data_given_orientation == 'COLUMN': ANSWER_DATA = ANSWER_DATA.transpose()
                            ANSWER_DATA = np.insert(ANSWER_DATA, col_idx, given_value_to_insert,
                                                    axis=0 if data_given_orientation == 'COLUMN' else 1)

                            # DONT ZIP ANSWER TO SPARSE DICT, RETURNED DATA WILL BE UNZIPPED TO NP ARRAY IF SPARSE DICT THEN TESTED AGAINST KEY

                            if header_is_given:
                                if header_string_is_given:
                                    ANSWER_HEADER = np.insert(PRIMAL_HEADER.copy(), col_idx, ['GIVEN INTERCEPT'], axis=1)
                                elif not header_string_is_given:
                                    ANSWER_HEADER = np.insert(PRIMAL_HEADER.copy(), col_idx, 'INTERCEPT', axis=1)
                            elif not header_is_given:
                                ANSWER_HEADER = None

                            # END ANSWER KEY #######################################################################################################






                            trial_description = \
                                f'\ndata_given_format = {data_given_format}' \
                                f'\ndata_given_orientation = {data_given_orientation}' \
                                f'\ncol_idx = {col_idx}' \
                                f'\nvalue_to_insert = {given_value_to_insert}' \
                                f'\nheader_string = {given_header_string}'


                            # TEST ACT & ANSWER DATA CONGRUENCE
                            if not np.array_equiv(ANSWER_DATA, ACT_DATA):
                                print_red(f'\nTrial description:')
                                print_red(trial_description)
                                print_red(f'\nANSWER DATA AND ACTUAL DATA ARE NOT CONGRUENT')
                                print_red(f'ANSWER DATA = ')
                                print_red(ANSWER_DATA)
                                print()
                                print_red(f'ACTUAL DATA = ')
                                print_red(ACT_DATA)
                                wls.winlinsound(222, 3)
                                raise Exception(f'FAIL.')

                            # TEST ACT & ANSWER HEADER CONGRUENCE
                            if not np.array_equiv(ANSWER_HEADER, ACT_HEADER):
                                print_red(f'\nTrial description:')
                                print_red(trial_description)
                                print_red(f'\nANSWER HEADER AND ACTUAL HEADER ARE NOT CONGRUENT')
                                print_red(f'ANSWER HEADER = ')
                                print_red(ANSWER_HEADER)
                                print()
                                print_red(f'ACTUAL HEADER = ')
                                print_red(ACT_HEADER)
                                wls.winlinsound(222, 3)
                                raise Exception(f'FAIL.')



    # IF REACH THIS POINT, PERMUTATION TESTS PASSED, SO IF fails==0, ALL TESTS PASSED
    if fails == 0:
        print_green(f'*** ALL TESTS PASSED ***')
        for _ in range(3): wls.winlinsound(888, 1); time.sleep(1)
    else:
        print_red(f'*** PERMUTATION TESTS PASSED, BUT {fails} VALIDATION TESTS FAILED ***'); wls.winlinsound(222, 3)



















