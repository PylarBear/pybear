import numpy as np
import sparse_dict as sd
import sys
from debug import get_module_name as gmn
from MLObjects.ObjectOrienter import MLObjectOrienter as mloo
from general_data_ops import create_random_sparse_numpy as crsn


# A MODULE TO VALIDATE FUNCTIONALITY OF MLObjectOrienter W/ XTX &/OR XTX_INV AS INPUTS, AND OUTPUTS OF XTX &/OR XTX_INV
# VERIFIED TO WORK FOR XTX & XTX_INV INPUTS AND XTX & XTX_INV OUTPUTS 12/18/22
# INPUTS OF DATA, DATA_TRANSPOSE, TARGET, TARGET_TRANSPOSE, TARGET_AS_LIST HANDLED IN OTHER MODULES TO MANAGE COMPLEXITY




def _error(words):
    raise Exception(f'\n*** {words} ***\n')




calling_module = gmn.get_module_name(str(sys.modules[__name__]))
calling_fxn = 'guard_test'

_rows = 5
_cols = 4

try_ctr = 0
while True:    # KEEP LOOPING UNTIL MAKE A MATRIX THAT CAN BE INVERTED
    try_ctr += 1
    print(f'\nAttempt #{try_ctr} to build an invertible test matrix...')
    DATA_NP_TEMPLATE = crsn.create_random_sparse_numpy(1,10,(_rows,_cols),50,np.float64)
    TEST_XTX = np.matmul(DATA_NP_TEMPLATE.transpose(), DATA_NP_TEMPLATE)
    try:
        np.linalg.inv(TEST_XTX)
        del TEST_XTX, try_ctr
        print(f'Successful.')
        break
    except:
        print(f'failed.')
        continue



XTX_NP_TEMPLATE = np.matmul(DATA_NP_TEMPLATE.transpose(), DATA_NP_TEMPLATE)
XTX_SD_TEMPLATE = sd.zip_list_as_py_float(XTX_NP_TEMPLATE)

XTX_INV_NP_TEMPLATE = np.linalg.inv(XTX_NP_TEMPLATE)
XTX_INV_SD_TEMPLATE = sd.zip_list_as_py_float(XTX_INV_NP_TEMPLATE)

# FALSE_XTX_INV
try_ctr = 0
while True:    # KEEP LOOPING UNTIL MAKE A MATRIX THAT CAN BE INVERTED
    try_ctr += 1
    print(f'\nAttempt #{try_ctr} to build an invertible test false xtx_inv matrix...')
    FALSE_XTX_INV_NP_TEMPLATE = crsn.create_random_sparse_numpy(1,10,(_cols,_cols),50,np.float64)
    TEST_XTX = np.matmul(FALSE_XTX_INV_NP_TEMPLATE.transpose(), FALSE_XTX_INV_NP_TEMPLATE)
    try:
        np.linalg.inv(TEST_XTX)
        del TEST_XTX, try_ctr
        print(f'Successful.')
        break
    except:
        print(f'failed.')
        continue

FALSE_XTX_INV_SD_TEMPLATE = sd.zip_list_as_py_float(FALSE_XTX_INV_NP_TEMPLATE)


MASTER_BYPASS_VALIDATION = [True, False]
MASTER_XTX_OBJECTS = [None, XTX_NP_TEMPLATE, XTX_SD_TEMPLATE]             # 12/15/22 PROVED THAT MLObjectOrienter CATCHES DELIBERATELY WRONG XTX_INV
MASTER_XTX_INV_OBJECTS = [XTX_INV_NP_TEMPLATE, XTX_INV_SD_TEMPLATE, None] #, FALSE_XTX_INV_NP_TEMPLATE, FALSE_XTX_INV_SD_TEMPLATE]
MASTER_XTX_RETURN_FORMATS = ['ARRAY', 'SPARSE_DICT', 'AS_GIVEN']
MASTER_XTX_INV_RETURN_FORMATS = ['ARRAY', 'SPARSE_DICT', 'AS_GIVEN']
MASTER_RETURN_OBJECTS = [[], ['XTX'], ['XTX_INV'], ['XTX', 'XTX_INV']]



total_trials = np.product(list(map(len, [MASTER_BYPASS_VALIDATION, MASTER_XTX_OBJECTS, MASTER_XTX_INV_OBJECTS,
                        MASTER_XTX_RETURN_FORMATS, MASTER_XTX_INV_RETURN_FORMATS, MASTER_RETURN_OBJECTS])))


ctr = 0
for bypass_validation in MASTER_BYPASS_VALIDATION:
    for XTX_OBJECT in MASTER_XTX_OBJECTS:
        for XTX_INV_OBJECT in MASTER_XTX_INV_OBJECTS:
            for xtx_return_format in MASTER_XTX_RETURN_FORMATS:
                for xtx_inv_return_format in MASTER_XTX_INV_RETURN_FORMATS:
                    for DUM_RETURN_OBJECTS in MASTER_RETURN_OBJECTS:

                        if XTX_OBJECT is None and XTX_INV_OBJECT is None:
                            total_trials -= 1
                            continue

                        ctr += 1
                        print(f'*'*90)
                        print(f'Running trial {ctr} of at most {total_trials}...')

                        exp_calling_module = calling_module
                        exp_calling_fxn = calling_fxn

                        exp_data_given_orientation = None
                        exp_data_given_format = None
                        exp_data_current_orientation = None
                        exp_data_current_format = None
                        exp_data_return_orientation = None
                        exp_data_return_format = None
                        exp_data_transpose_given_orientation = None
                        exp_data_transpose_given_format = None
                        exp_data_transpose_current_orientation = None
                        exp_data_transpose_current_format = None
                        exp_data_transpose_return_orientation = None
                        exp_data_transpose_return_format = None
                        EXP_DATA = None
                        EXP_DATA_TRANSPOSE = None
                        EXP_TARGET = None
                        EXP_TARGET_TRANSPOSE = None
                        EXP_TARGET_AS_LIST = None


                        exp_xtx_given_format = 'SPARSE_DICT' if isinstance(XTX_OBJECT, dict) else 'ARRAY' if isinstance(XTX_OBJECT, np.ndarray) else None if XTX_OBJECT is None else _error(f'xtx_given_format')
                        exp_xtx_inv_given_format = 'SPARSE_DICT' if isinstance(XTX_INV_OBJECT, dict) else 'ARRAY' if isinstance(XTX_INV_OBJECT, np.ndarray) else None if XTX_INV_OBJECT is None else _error(f'xtx_inv_given_format')

                        if xtx_return_format != 'AS_GIVEN':
                            exp_xtx_current_format = xtx_return_format
                            exp_xtx_return_format = xtx_return_format
                        elif xtx_return_format == 'AS_GIVEN':
                            exp_xtx_current_format = exp_xtx_given_format if not exp_xtx_given_format is None else exp_xtx_inv_given_format
                            exp_xtx_return_format = exp_xtx_given_format if not exp_xtx_given_format is None else exp_xtx_inv_given_format

                        if 'XTX' in DUM_RETURN_OBJECTS:
                            if not XTX_OBJECT is None:
                                EXP_XTX = XTX_OBJECT
                            else:
                                if exp_xtx_inv_given_format == 'ARRAY':
                                    EXP_XTX = np.linalg.inv(XTX_INV_OBJECT)
                                elif exp_xtx_inv_given_format == 'SPARSE_DICT':
                                    EXP_XTX = np.linalg.inv(sd.unzip_to_ndarray_float64(XTX_INV_OBJECT)[0])
                        elif 'XTX' not in DUM_RETURN_OBJECTS:
                            EXP_XTX = None
                            exp_xtx_current_format = None
                            exp_xtx_return_format = None

                        if isinstance(EXP_XTX, np.ndarray) and exp_xtx_return_format == 'SPARSE_DICT':
                            EXP_XTX = sd.zip_list_as_py_float(EXP_XTX)
                        elif isinstance(EXP_XTX, dict) and exp_xtx_return_format == 'ARRAY':
                            EXP_XTX = sd.unzip_to_ndarray_float64(EXP_XTX)[0]


                        if xtx_inv_return_format != 'AS_GIVEN':
                            exp_xtx_inv_current_format = xtx_inv_return_format
                            exp_xtx_inv_return_format = xtx_inv_return_format
                        elif xtx_inv_return_format == 'AS_GIVEN':
                            exp_xtx_inv_current_format = exp_xtx_inv_given_format if not exp_xtx_inv_given_format is None else exp_xtx_given_format
                            exp_xtx_inv_return_format = exp_xtx_inv_given_format if not exp_xtx_inv_given_format is None else exp_xtx_given_format

                        if 'XTX_INV' in DUM_RETURN_OBJECTS:
                            if not XTX_INV_OBJECT is None:
                                EXP_XTX_INV = XTX_INV_OBJECT
                            else:
                                if exp_xtx_given_format == 'ARRAY':
                                    EXP_XTX_INV = np.linalg.inv(XTX_OBJECT)
                                elif exp_xtx_given_format == 'SPARSE_DICT':
                                    EXP_XTX_INV = np.linalg.inv(sd.unzip_to_ndarray_float64(XTX_OBJECT)[0])
                        if 'XTX_INV' not in DUM_RETURN_OBJECTS:
                            EXP_XTX_INV = None
                            exp_xtx_inv_current_format = None
                            exp_xtx_inv_return_format = None

                        if isinstance(EXP_XTX_INV, np.ndarray) and exp_xtx_inv_return_format == 'SPARSE_DICT':
                            EXP_XTX_INV = sd.zip_list_as_py_float(EXP_XTX_INV)
                        elif isinstance(EXP_XTX_INV, dict) and exp_xtx_inv_return_format == 'ARRAY':
                            EXP_XTX_INV = sd.unzip_to_ndarray_float64(EXP_XTX_INV)[0]

                        print(f'XTX is {"" if not XTX_OBJECT is None else "not "}given, XTX_INV is {"" if not XTX_INV_OBJECT is None else "not "}given.')
                        print(f'Expected output:\n',
                              f'exp_this_module = MLObjectOrienter\n',
                              f'exp_calling_module = {exp_calling_module}\n',
                              f'exp_calling_fxn = {exp_calling_fxn}\n',
                              f'exp_data_given_orientation = {exp_data_given_orientation}\n',
                              f'exp_data_given_format = {exp_data_given_format}\n',
                              f'exp_data_current_orientation = {exp_data_current_orientation}\n',
                              f'exp_data_current_format = {exp_data_current_format}\n'
                              f'exp_data_return_orientation = {exp_data_return_orientation}\n',
                              f'exp_data_return_format = {exp_data_return_format}\n',
                              f'exp_data_transpose_given_orientation = {exp_data_transpose_given_orientation}\n',
                              f'exp_data_transpose_given_format = {exp_data_transpose_given_format}\n',
                              f'exp_data_transpose_current_orientation = {exp_data_transpose_current_orientation}\n',
                              f'exp_data_transpose_current_format = {exp_data_transpose_current_format}\n'
                              f'exp_data_transpose_return_orientation = {exp_data_transpose_return_orientation}\n',
                              f'exp_data_transpose_return_format = {exp_data_transpose_return_format}\n'
                              f'exp_xtx_given_format = {exp_xtx_given_format}\n'
                              f'exp_xtx_return_format = {exp_xtx_return_format}\n'
                              f'exp_xtx_current_format = {exp_xtx_current_format}\n'
                              f'exp_xtx_inv_given_format = {exp_xtx_inv_given_format}\n'
                              f'exp_xtx_inv_return_format = {exp_xtx_inv_return_format}\n'
                              f'exp_xtx_inv_current_format = {exp_xtx_inv_current_format}\n'
                              f'EXP_XTX = {EXP_XTX}\n',
                              f'EXP_XTX_INV = {EXP_XTX_INV}\n',
                              f'DUM_RETURN_OBJECTS = {DUM_RETURN_OBJECTS}'
                              )


                        Dummy = mloo.MLObjectOrienter(
                            DATA=None,
                            # ENTER format & orientation for DATA AS IT WOULD BE IF DATA IS GIVEN, EVEN
                            # IF IS NOT GIVEN, IF A DATA OBJECT IS TO BE RETURNED
                            # DATA given_orientation MUST BE ENTERED IF ANY DATA OBJECT IS TO BE RETURNED
                            data_given_orientation=None,
                            data_return_orientation=None,
                            data_return_format=None,

                            DATA_TRANSPOSE=None,
                            data_transpose_return_orientation=None,
                            data_transpose_return_format=None,

                            XTX=XTX_OBJECT,
                            xtx_return_format=xtx_return_format,  # XTX IS SYMMETRIC SO DONT NEED ORIENTATION

                            XTX_INV=XTX_INV_OBJECT,
                            xtx_inv_return_format=xtx_inv_return_format,  # XTX IS SYMMETRIC SO DONT NEED ORIENTATION

                            ############################################################################################################
                            TARGET=None,
                            # ENTER format & orientation for TARGET AS IT WOULD BE IF TARGET IS GIVEN,
                            # EVEN IF IS NOT GIVEN, IF A TARGET OBJECT IS TO BE RETURNED
                            # TARGET given_orientation MUST BE ENTERED IF ANY TARGET OBJECT IS TO BE RETURNED
                            target_given_orientation=None,
                            target_return_orientation=None, #'AS_GIVEN',
                            target_return_format=None, #'AS_GIVEN',
                            target_is_multiclass=None, #False,
                            TARGET_TRANSPOSE=None,
                            target_transpose_return_format=None, #'AS_GIVEN',
                            TARGET_AS_LIST=None,
                            ############################################################################################################

                            RETURN_OBJECTS=DUM_RETURN_OBJECTS,
                            bypass_validation=bypass_validation,
                            calling_module=calling_module,
                            calling_fxn=calling_fxn)


                        ACT_DATA = Dummy.DATA
                        ACT_DATA_TRANSPOSE = Dummy.DATA_TRANSPOSE
                        ACT_XTX = Dummy.XTX
                        ACT_XTX_INV = Dummy.XTX_INV

                        ACT_TARGET = Dummy.TARGET
                        ACT_TARGET_TRANSPOSE = Dummy.TARGET_TRANSPOSE
                        ACT_TARGET_AS_LIST = Dummy.TARGET_AS_LIST

                        act_this_module = 'MLObjectOrienter'
                        act_calling_module = gmn.get_module_name(str(sys.modules[__name__]))
                        act_calling_fxn = 'guard_test'

                        act_data_given_orientation = Dummy.data_given_orientation
                        act_data_given_format = Dummy.data_given_format
                        act_data_current_orientation = Dummy.data_return_orientation
                        act_data_current_format = Dummy.data_current_format
                        act_data_return_orientation = Dummy.data_return_orientation
                        act_data_return_format = Dummy.data_return_format

                        act_data_transpose_given_orientation = Dummy.data_transpose_given_orientation
                        act_data_transpose_given_format = Dummy.data_transpose_given_format
                        act_data_transpose_current_orientation = Dummy.data_transpose_current_orientation
                        act_data_transpose_current_format = Dummy.data_transpose_current_format
                        act_data_transpose_return_orientation = Dummy.data_transpose_return_orientation
                        act_data_transpose_return_format = Dummy.data_transpose_return_format

                        act_xtx_given_format = Dummy.xtx_given_format
                        act_xtx_return_format = Dummy.xtx_return_format
                        act_xtx_current_format = Dummy.xtx_current_format
                        act_xtx_inv_given_format = Dummy.xtx_inv_given_format
                        act_xtx_inv_return_format = Dummy.xtx_inv_return_format
                        act_xtx_inv_current_format = Dummy.xtx_inv_current_format


                        NAMES = [
                            'calling_module',
                            'calling_fxn',
                            'data_given_orientation',
                            'data_given_format',
                            'data_current_orientation',
                            'data_current_format',
                            'data_return_orientation',
                            'data_return_format',
                            'data_transpose_given_orientation',
                            'data_transpose_given_format',
                            'data_transpose_current_orientation',
                            'data_transpose_current_format',
                            'data_transpose_return_orientation',
                            'data_transpose_return_format',
                            'DATA',
                            'DATA_TRANSPOSE',
                            'xtx_given_format',
                            'xtx_return_format',
                            'xtx_current_format',
                            'XTX',
                            'xtx_inv_given_format',
                            'xtx_inv_return_format',
                            'xtx_inv_current_format',
                            'XTX_INV',
                            'TARGET',
                            'TARGET_TRANSPOSE',
                            'TARGET_AS_LIST'
                        ]

                        EXPECTED_INPUTS = [
                            exp_calling_module,
                            exp_calling_fxn,
                            exp_data_given_orientation,
                            exp_data_given_format,
                            exp_data_current_orientation,
                            exp_data_current_format,
                            exp_data_return_orientation,
                            exp_data_return_format,
                            exp_data_transpose_given_orientation,
                            exp_data_transpose_given_format,
                            exp_data_transpose_current_orientation,
                            exp_data_transpose_current_format,
                            exp_data_transpose_return_orientation,
                            exp_data_transpose_return_format,
                            EXP_DATA,
                            EXP_DATA_TRANSPOSE,
                            exp_xtx_given_format,
                            exp_xtx_return_format,
                            exp_xtx_current_format,
                            EXP_XTX,
                            exp_xtx_inv_given_format,
                            exp_xtx_inv_return_format,
                            exp_xtx_inv_current_format,
                            EXP_XTX_INV,
                            EXP_TARGET,
                            EXP_TARGET_TRANSPOSE,
                            EXP_TARGET_AS_LIST
                        ]

                        ACTUAL_OUTPUTS = [
                            act_calling_module,
                            act_calling_fxn,
                            act_data_given_orientation,
                            act_data_given_format,
                            act_data_current_orientation,
                            act_data_current_format,
                            act_data_return_orientation,
                            act_data_return_format,
                            act_data_transpose_given_orientation,
                            act_data_transpose_given_format,
                            act_data_transpose_current_orientation,
                            act_data_transpose_current_format,
                            act_data_transpose_return_orientation,
                            act_data_transpose_return_format,
                            ACT_DATA,
                            ACT_DATA_TRANSPOSE,
                            act_xtx_given_format,
                            act_xtx_return_format,
                            act_xtx_current_format,
                            ACT_XTX,
                            act_xtx_inv_given_format,
                            act_xtx_inv_return_format,
                            act_xtx_inv_current_format,
                            ACT_XTX_INV,
                            ACT_TARGET,
                            ACT_TARGET_TRANSPOSE,
                            ACT_TARGET_AS_LIST
                        ]

                        for description, expected_thing, actual_thing in zip(NAMES, EXPECTED_INPUTS, ACTUAL_OUTPUTS):

                            try:
                                is_equal = np.array_equiv(expected_thing, actual_thing)
                                # print(f'\033[91m\n*** TEST EXCEPTED ON np.array_equiv METHOD ***\033[0m\x1B[0m\n')
                            except:
                                try:
                                    is_equal = expected_thing == actual_thing
                                except:
                                    print(f'{description}:')
                                    print(f'\n\033[91mEXP_OBJECT = \n{expected_thing}\033[0m\x1B[0m\n')
                                    print(f'\n\033[91mACT_OBJECT = \n{actual_thing}\033[0m\x1B[0m\n')
                                    raise Exception(f'\n*** TEST FAILED "==" METHOD ***\n')

                            if not is_equal:
                                print(f'\n\033[91mEXP_OBJECT = \n{expected_thing}\033[0m\x1B[0m\n')
                                print(f'\n\033[91mACT_OBJECT = \n{actual_thing}\033[0m\x1B[0m\n')
                                raise Exception(
                                    f'\n*** {description} FAILED EQUALITY TEST, \nexpected = \n{expected_thing}\n'
                                    f'actual = \n{actual_thing} ***\n')
                            else:
                                pass  # print(f'\033[92m     *** {description} PASSED ***\033[0m\x1B[0m')

print(f'\n\033[92m*** TEST DONE. ALL PASSED. ***\033[0m\x1B[0m\n')


















