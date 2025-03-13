import numpy as np
import sparse_dict as sd
import sys, time
from debug import get_module_name as gmn
from MLObjects.ObjectOrienter import MLObjectOrienter as mloo


# A MODULE TO VALIDATE FUNCTIONALITY OF MLObjectOrienter W/ TARGET, TARGET_TRANSPOSE &/OR TARGET_AS_LIST AS INPUTS, AND OUTPUTS OF
# TARGET, TARGET_TRANSPOSE &/OR TARGET_AS_LIST

# THIS TEST TAKES 9 HRS TO RUN :(  VERIFIED THIS TEST CODE IS GOOD AND MLObjectOrienter PASSES 12/18/22.


# INPUTS OF DATA, DATA_TRANSPOSE, XTX, & XTX_INV HANDLED IN OTHER MODULES TO MANAGE COMPLEXITY


def _error(words):
    raise Exception(f'\n*** {words} ***\n')




calling_module = gmn.get_module_name(str(sys.modules[__name__]))
calling_fxn = 'guard_test'


CNSCT = np.random.randint(0,10,(1,10))       # CORRECT NUMPY SINGLE_CLASS_TEMPLATE
CNMCT = np.random.randint(0,10,(3,10))       # CORRECT NUMPY MULTI_CLASS_TEMPLATE
WNSCT = np.random.randint(0,10,(1,10))       # WRONG_NUMPY SINGLE_CLASS_TEMPLATE
WNMCT = np.random.randint(0,10,(3,10))       # WRONG_NUMPY MULTI_CLASS_TEMPLATE


def to_dict(ARG):
    if isinstance(ARG, dict): pass
    elif isinstance(ARG, np.ndarray): return sd.zip_list_as_py_float(ARG)

def transposer(ARG):
    if isinstance(ARG, np.ndarray): return ARG.transpose()
    elif isinstance(ARG, dict): return sd.sparse_transpose(ARG)

def to_list(ARG):
    if isinstance(ARG, dict): return sd.unzip_to_ndarray_float64(ARG)[0]
    elif isinstance(ARG, np.ndarray): pass

def is_multiclass(ARG):
    if isinstance(ARG, np.ndarray):
        if len(ARG) == 1 or len(ARG[0]) == 1: return False
        else: return True
    elif isinstance(ARG, dict):
        if sd.outer_len(ARG) == 1 or sd.inner_len_quick(ARG) == 1: return False
        else: return True


MASTER_BYPASS_VALIDATION = [True, False]

MASTER_MULTICLASS =              [False, False, True]
MASTER_SINGLE_DOUBLE =           ['SINGLE', 'DOUBLE', 'DOUBLE']

# MASTER_TAR_OBJECTS =             ['build_in_process', None]
# MASTER_TAR_GIVEN_FORMAT =        ['ARRAY', 'SPARSE_DICT']
# MASTER_TAR_GIVEN_ORIENTATION =   ['COLUMN','ROW']
MASTER_TAR_RETURN_FORMAT =       ['ARRAY', 'SPARSE_DICT', 'AS_GIVEN']
MASTER_TAR_RETURN_ORIENTATION =  ['COLUMN', 'ROW', 'AS_GIVEN']

MASTER_TAR_OBJECTS =           ['build_in_process', 'build_in_process', 'build_in_process', 'build_in_process', None]
MASTER_TAR_GIVEN_FORMAT =      [           'ARRAY',            'ARRAY',      'SPARSE_DICT',      'SPARSE_DICT', None]
MASTER_TAR_GIVEN_ORIENTATION = [          'COLUMN',              'ROW',           'COLUMN',              'ROW', None]

# MASTER_TAR_T_OBJECTS =           ['build_in_process', None]
# MASTER_TAR_T_GIVEN_FORMAT =      ['ARRAY', 'SPARSE_DICT']
# MASTER_TAR_T_GIVEN_ORIENTATION = ['COLUMN','ROW']
MASTER_TAR_T_RETURN_FORMAT =     ['ARRAY', 'SPARSE_DICT', 'AS_GIVEN']
MASTER_TAR_T_RETURN_ORIENTATION =['COLUMN', 'ROW', 'AS_GIVEN']

MASTER_TAR_T_OBJECTS =           ['build_in_process', 'build_in_process', 'build_in_process', 'build_in_process', None]
MASTER_TAR_T_GIVEN_FORMAT =      [           'ARRAY',            'ARRAY',      'SPARSE_DICT',      'SPARSE_DICT', None]
MASTER_TAR_T_GIVEN_ORIENTATION = [          'COLUMN',              'ROW',           'COLUMN',              'ROW', None]

# 12/16/22 PROVED MLObjectOrienter CAN FIND WHEN TARGET & TARGET TRANSPOSE ROW/COLUMN/NOT_T/T ARE INCONGRUOUS


# MASTER_T_A_L_OBJECTS =           ['build_in_process', None]
# MASTER_T_A_L_GIVEN_FORMAT =      ['ARRAY', 'SPARSE_DICT']
# MASTER_T_A_L_GIVEN_ORIENTATION = ['COLUMN','ROW']
MASTER_T_A_L_RETURN_ORIENTATION =['COLUMN', 'ROW', 'AS_GIVEN']

MASTER_T_A_L_OBJECTS =           ['build_in_process', 'build_in_process', None] #, 'build_in_process', 'build_in_process'] <-- USE THIS TO VALIDATE EXCEPTS IF GIVEN T_A_L IS DICT
MASTER_T_A_L_GIVEN_FORMAT =      [           'ARRAY',            'ARRAY', None] #,      'SPARSE_DICT', '     SPARSE_DICT'] <-- USE THIS TO VALIDATE EXCEPTS IF GIVEN T_A_L IS DICT
MASTER_T_A_L_GIVEN_ORIENTATION = [          'COLUMN',              'ROW', None] #,           'COLUMN',              'ROW'] <-- USE THIS TO VALIDATE EXCEPTS IF GIVEN T_A_L IS DICT

MASTER_RETURN_OBJECTS = [[], ['TARGET'], ['TARGET_TRANSPOSE'], ['TARGET_AS_LIST'],
                         ['TARGET', 'TARGET_TRANSPOSE'], ['TARGET', 'TARGET_AS_LIST'], ['TARGET_TRANSPOSE', 'TARGET_AS_LIST'],
                         ['TARGET', 'TARGET_TRANSPOSE', 'TARGET_AS_LIST']]

# total_trials = np.product(list(map(len, [MASTER_BYPASS_VALIDATION, MASTER_MULTICLASS, MASTER_TAR_OBJECTS,
#              MASTER_TAR_GIVEN_FORMAT, MASTER_TAR_GIVEN_ORIENTATION, MASTER_TAR_RETURN_FORMAT, MASTER_TAR_RETURN_ORIENTATION,
#              MASTER_TAR_T_OBJECTS, MASTER_TAR_T_GIVEN_FORMAT, MASTER_TAR_T_GIVEN_ORIENTATION, MASTER_TAR_T_RETURN_FORMAT,
#              MASTER_TAR_T_RETURN_ORIENTATION, MASTER_T_A_L_OBJECTS, MASTER_T_A_L_GIVEN_FORMAT, MASTER_T_A_L_GIVEN_ORIENTATION,
#              MASTER_T_A_L_RETURN_ORIENTATION, MASTER_RETURN_OBJECTS])))

total_trials = np.product(list(map(len, [MASTER_BYPASS_VALIDATION, MASTER_MULTICLASS,
                                         MASTER_TAR_OBJECTS, MASTER_TAR_RETURN_FORMAT, MASTER_TAR_RETURN_ORIENTATION,
                                         MASTER_TAR_T_OBJECTS, MASTER_TAR_T_RETURN_FORMAT, MASTER_TAR_T_RETURN_ORIENTATION,
                                         MASTER_T_A_L_OBJECTS, MASTER_T_A_L_RETURN_ORIENTATION,
                                         MASTER_RETURN_OBJECTS])))

print(f'\nMaximum total trials = {total_trials:,}\n')
time.sleep(1)

exp_calling_module = calling_module
exp_calling_fxn = calling_fxn
EXP_DATA = None
exp_data_given_orientation = None
exp_data_given_format = None
exp_data_current_orientation = None
exp_data_current_format = None
exp_data_return_orientation = None
exp_data_return_format = None
EXP_DATA_TRANSPOSE = None
exp_data_transpose_given_orientation = None
exp_data_transpose_given_format = None
exp_data_transpose_current_orientation = None
exp_data_transpose_current_format = None
exp_data_transpose_return_orientation = None
exp_data_transpose_return_format = None
EXP_XTX = None
exp_xtx_given_format = None
exp_xtx_return_format = None
exp_xtx_current_format = None
EXP_XTX_INV = None
exp_xtx_inv_given_format = None
exp_xtx_inv_return_format = None
exp_xtx_inv_current_format = None


# for bypass_validation in MASTER_BYPASS_VALIDATION:
#     for is_multiclass, single_double in zip(MASTER_MULTICLASS, MASTER_SINGLE_DOUBLE):
#         for GIVEN_TARGET_OBJECT in MASTER_TAR_OBJECTS:
#             for target_given_format in MASTER_TAR_GIVEN_FORMAT:
#                 for target_return_format in MASTER_TAR_RETURN_FORMAT:
#                     for target_given_orientation in MASTER_TAR_GIVEN_ORIENTATION:
#                         for target_return_orientation in MASTER_TAR_RETURN_ORIENTATION:
#                             for GIVEN_TARGET_TRANSPOSE_OBJECT in MASTER_TAR_T_OBJECTS:
#                                 for target_transpose_given_format in MASTER_TAR_T_GIVEN_FORMAT:
#                                     for target_transpose_given_orientation in MASTER_TAR_T_GIVEN_ORIENTATION:
#                                         for target_transpose_return_format in MASTER_TAR_T_RETURN_FORMAT:
#                                             for target_transpose_return_orientation in MASTER_TAR_T_RETURN_ORIENTATION:
#                                                 for GIVEN_TARGET_AS_LIST_OBJECT in MASTER_T_A_L_OBJECTS:
#                                                     for target_as_list_given_format in MASTER_T_A_L_GIVEN_FORMAT:
#                                                         for target_as_list_given_orientation in MASTER_T_A_L_GIVEN_ORIENTATION:
#                                                             for target_as_list_return_orientation in MASTER_T_A_L_RETURN_ORIENTATION:
#                                                                 for RETURN_OBJECTS in MASTER_RETURN_OBJECTS:






ctr = 0
for bypass_validation in MASTER_BYPASS_VALIDATION:
    for is_multiclass, single_double in zip(MASTER_MULTICLASS, MASTER_SINGLE_DOUBLE):
        for GIVEN_TARGET_OBJECT, target_given_format, target_given_orientation in zip(MASTER_TAR_OBJECTS, MASTER_TAR_GIVEN_FORMAT, MASTER_TAR_GIVEN_ORIENTATION):
            for target_return_format in MASTER_TAR_RETURN_FORMAT:
                for target_return_orientation in MASTER_TAR_RETURN_ORIENTATION:
                    for GIVEN_TARGET_TRANSPOSE_OBJECT, target_transpose_given_format, target_transpose_given_orientation in zip(MASTER_TAR_T_OBJECTS, MASTER_TAR_T_GIVEN_FORMAT, MASTER_TAR_T_GIVEN_ORIENTATION):
                        for target_transpose_return_format in MASTER_TAR_T_RETURN_FORMAT:
                            for target_transpose_return_orientation in MASTER_TAR_T_RETURN_ORIENTATION:
                                for GIVEN_TARGET_AS_LIST_OBJECT, target_as_list_given_format, target_as_list_given_orientation in zip(MASTER_T_A_L_OBJECTS, MASTER_T_A_L_GIVEN_FORMAT, MASTER_T_A_L_GIVEN_ORIENTATION):
                                    for target_as_list_return_orientation in MASTER_T_A_L_RETURN_ORIENTATION:
                                        for RETURN_OBJECTS in MASTER_RETURN_OBJECTS:

                                            if GIVEN_TARGET_OBJECT is None and GIVEN_TARGET_TRANSPOSE_OBJECT is None and GIVEN_TARGET_AS_LIST_OBJECT is None and \
                                                    ('TARGET' in RETURN_OBJECTS or 'TARGET_TRANSPOSE' or 'TARGET_AS_LIST' in RETURN_OBJECTS):
                                                total_trials -= 1
                                                continue

                                            # BEAR NOTES
                                            # TO SEE MORE PRINTS TO SCREEN, CHANGE MODULUS IN BELOW EQUATION
                                            # TO PRINT EXPECTED OUTPUTS TO SCREEN UNCOMMENT THE "EXPECTED OUTPUTS" SECTION

                                            ctr += 1
                                            if ctr % 2000 == 0:
                                                print(f'*'*90)
                                                print(f'Running trial {ctr:,} of at most {total_trials:,}...')

                                                print(f'TARGET is {"not given" if GIVEN_TARGET_OBJECT is None else "given"}')
                                                print(f'TARGET_TRANSPOSE is {"not given" if GIVEN_TARGET_TRANSPOSE_OBJECT is None else "given"}')
                                                print(f'TARGET_AS_LIST is {"not given" if GIVEN_TARGET_AS_LIST_OBJECT is None else "given"}')

                                                print(f'RETURN_OBJECTS = ')
                                                print(RETURN_OBJECTS)

                                            ########################################################################################################################
                                            ########################################################################################################################
                                            ## MODIFY TARGET TEMPLATE TO DESIRED INPUTS #############################################################################

                                            # BUILD BASE TEMPLATE TO BUILD ALL TARGET OBJECTS FROM ########################
                                            exp_is_multiclass = is_multiclass
                                            if not exp_is_multiclass: BASE_TARGET_OBJECT = CNSCT
                                            elif exp_is_multiclass: BASE_TARGET_OBJECT = CNMCT
                                            # END BUILD BASE TEMPLATE TO BUILD ALL TARGET OBJECTS FROM ####################

                                            # TARGET ######################################################################
                                            if GIVEN_TARGET_OBJECT is None:
                                                target_given_format = None
                                                target_given_orientation = None
                                            elif not GIVEN_TARGET_OBJECT is None:
                                                GIVEN_TARGET_OBJECT = BASE_TARGET_OBJECT
                                                if target_given_orientation == 'ROW':
                                                    GIVEN_TARGET_OBJECT = transposer(GIVEN_TARGET_OBJECT)

                                                if target_given_format == 'ARRAY': pass
                                                elif target_given_format == 'SPARSE_DICT':
                                                    GIVEN_TARGET_OBJECT = to_dict(GIVEN_TARGET_OBJECT)

                                                if single_double == 'SINGLE' and len(GIVEN_TARGET_OBJECT) == 1:
                                                    GIVEN_TARGET_OBJECT = GIVEN_TARGET_OBJECT[0]
                                            # END TARGET ######################################################################

                                            # TARGET_TRANSPOSE ######################################################################
                                            if GIVEN_TARGET_TRANSPOSE_OBJECT is None:
                                                target_transpose_given_format = None
                                                target_transpose_given_orientation = None
                                            elif not GIVEN_TARGET_TRANSPOSE_OBJECT is None:
                                                GIVEN_TARGET_TRANSPOSE_OBJECT = BASE_TARGET_OBJECT
                                                if target_transpose_given_orientation == 'COLUMN':
                                                    GIVEN_TARGET_TRANSPOSE_OBJECT = transposer(GIVEN_TARGET_TRANSPOSE_OBJECT)
                                                # else DONT TRANSPOSE BECAUSE TARGET & TARGET_TRANSPOSE ORIENTATIONS ARE DIFFERENT

                                                if target_transpose_given_format == 'SPARSE_DICT':
                                                    GIVEN_TARGET_TRANSPOSE_OBJECT = to_dict(GIVEN_TARGET_TRANSPOSE_OBJECT)

                                                if single_double == 'SINGLE' and len(GIVEN_TARGET_TRANSPOSE_OBJECT) == 1:
                                                    GIVEN_TARGET_TRANSPOSE_OBJECT = GIVEN_TARGET_TRANSPOSE_OBJECT[0]
                                            # END TARGET_TRANSPOSE ######################################################################

                                            # TARGET_AS_LIST ######################################################################
                                            if GIVEN_TARGET_AS_LIST_OBJECT is None:
                                                target_as_list_given_format = None
                                                target_as_list_given_orientation = None
                                            elif not GIVEN_TARGET_AS_LIST_OBJECT is None:
                                                GIVEN_TARGET_AS_LIST_OBJECT = BASE_TARGET_OBJECT
                                                if target_as_list_given_orientation == 'ROW':
                                                    GIVEN_TARGET_AS_LIST_OBJECT = transposer(GIVEN_TARGET_AS_LIST_OBJECT)

                                                # ONLY HERE FOR TESTING OF EXCEPTION, MLObjectOrienter MUST EXCEPT IF GIVEN T_A_L AS DICT
                                                # VERIFIED TO WORK 5:32 PM 12/16/22
                                                # if target_as_list_given_format == 'ARRAY': pass
                                                # elif target_as_list_given_format == 'SPARSE_DICT':
                                                #     GIVEN_TARGET_AS_LIST_OBJECT = to_dict(GIVEN_TARGET_AS_LIST_OBJECT)

                                                if single_double == 'SINGLE' and len(GIVEN_TARGET_AS_LIST_OBJECT) == 1:
                                                    GIVEN_TARGET_AS_LIST_OBJECT = GIVEN_TARGET_AS_LIST_OBJECT[0]
                                            # END TARGET_AS_LIST ######################################################################

                                            ## END MODIFY TARGET TEMPLATE TO DESIRED INPUTS #############################################################################
                                            ########################################################################################################################
                                            ########################################################################################################################

                                            ########################################################################################################################
                                            ########################################################################################################################
                                            ## CREATE EXPECTED TARGET RETURN OBJECTS #############################################################################

                                            exp_target_given_format = target_given_format
                                            exp_target_given_orientation = target_given_orientation
                                            exp_target_transpose_given_format = target_transpose_given_format
                                            exp_target_transpose_given_orientation = target_transpose_given_orientation
                                            exp_target_as_list_given_format = target_as_list_given_format
                                            exp_target_as_list_given_orientation = target_as_list_given_orientation

                                            # TARGET ######################################################################
                                            if 'TARGET' not in RETURN_OBJECTS:
                                                EXP_TARGET = None
                                                exp_target_return_format = None
                                                exp_target_current_format = None
                                                exp_target_return_orientation = None
                                                exp_target_current_orientation = None
                                            elif 'TARGET' in RETURN_OBJECTS:
                                                EXP_TARGET = BASE_TARGET_OBJECT

                                                if target_return_format != 'AS_GIVEN':
                                                    exp_target_return_format = target_return_format
                                                    exp_target_current_format = target_return_format
                                                elif target_return_format == 'AS_GIVEN':
                                                    if not GIVEN_TARGET_OBJECT is None:
                                                        exp_target_return_format = exp_target_given_format
                                                        exp_target_current_format = exp_target_given_format
                                                    elif not GIVEN_TARGET_AS_LIST_OBJECT is None:
                                                        exp_target_return_format = exp_target_as_list_given_format
                                                        exp_target_current_format = exp_target_as_list_given_format
                                                    elif not GIVEN_TARGET_TRANSPOSE_OBJECT is None:
                                                        exp_target_return_format = exp_target_transpose_given_format
                                                        exp_target_current_format = exp_target_transpose_given_format
                                                    else: _error(f'TRYING TO BUILD TARGET BUT NO TARGET OBJECTS GIVEN')

                                                if target_return_orientation != 'AS_GIVEN':
                                                    exp_target_return_orientation = target_return_orientation
                                                    exp_target_current_orientation = target_return_orientation
                                                elif target_return_orientation == 'AS_GIVEN':
                                                    if not GIVEN_TARGET_OBJECT is None:
                                                        exp_target_return_orientation = exp_target_given_orientation
                                                        exp_target_current_orientation = exp_target_given_orientation
                                                    elif not GIVEN_TARGET_AS_LIST_OBJECT is None:
                                                        exp_target_return_orientation = exp_target_as_list_given_orientation
                                                        exp_target_current_orientation = exp_target_as_list_given_orientation
                                                    elif not GIVEN_TARGET_TRANSPOSE_OBJECT is None:
                                                        exp_target_return_orientation = exp_target_transpose_given_orientation
                                                        exp_target_current_orientation = exp_target_transpose_given_orientation
                                                    else: _error(f'TRYING TO BUILD TARGET BUT NO TARGET OBJECTS GIVEN')

                                                if exp_target_return_orientation == 'ROW':
                                                    EXP_TARGET = transposer(EXP_TARGET)

                                                if exp_target_return_format == 'ARRAY': pass
                                                elif exp_target_return_format == 'SPARSE_DICT':
                                                    EXP_TARGET = to_dict(EXP_TARGET)
                                            # END TARGET ######################################################################

                                            # TARGET_TRANSPOSE ######################################################################
                                            if 'TARGET_TRANSPOSE' not in RETURN_OBJECTS:
                                                EXP_TARGET_TRANSPOSE = None
                                                exp_target_transpose_return_format = None
                                                exp_target_transpose_current_format = None
                                                exp_target_transpose_return_orientation = None
                                                exp_target_transpose_current_orientation = None
                                            elif 'TARGET_TRANSPOSE' in RETURN_OBJECTS:
                                                EXP_TARGET_TRANSPOSE = BASE_TARGET_OBJECT

                                                if target_transpose_return_format != 'AS_GIVEN':
                                                    exp_target_transpose_return_format = target_transpose_return_format
                                                    exp_target_transpose_current_format = target_transpose_return_format
                                                elif target_transpose_return_format == 'AS_GIVEN':
                                                    if not GIVEN_TARGET_TRANSPOSE_OBJECT is None:
                                                        exp_target_transpose_return_format = exp_target_transpose_given_format
                                                        exp_target_transpose_current_format = exp_target_transpose_given_format
                                                    elif not GIVEN_TARGET_OBJECT is None:
                                                        exp_target_transpose_return_format = exp_target_given_format
                                                        exp_target_transpose_current_format = exp_target_given_format
                                                    elif not GIVEN_TARGET_AS_LIST_OBJECT is None:
                                                        exp_target_transpose_return_format = exp_target_as_list_given_format
                                                        exp_target_transpose_current_format = exp_target_as_list_given_format
                                                    else: _error(f'TRYING TO BUILD TARGET_TRANSPOSE BUT NO TARGET OBJECTS GIVEN')

                                                if target_transpose_return_orientation != 'AS_GIVEN':
                                                    exp_target_transpose_return_orientation = target_transpose_return_orientation
                                                    exp_target_transpose_current_orientation = target_transpose_return_orientation
                                                elif target_transpose_return_orientation == 'AS_GIVEN':
                                                    if not GIVEN_TARGET_TRANSPOSE_OBJECT is None:
                                                        exp_target_transpose_return_orientation = exp_target_transpose_given_orientation
                                                        exp_target_transpose_current_orientation = exp_target_transpose_given_orientation
                                                    elif not GIVEN_TARGET_OBJECT is None:
                                                        exp_target_transpose_return_orientation = exp_target_given_orientation
                                                        exp_target_transpose_current_orientation = exp_target_given_orientation
                                                    elif not GIVEN_TARGET_AS_LIST_OBJECT is None:
                                                        exp_target_transpose_return_orientation = exp_target_as_list_given_orientation
                                                        exp_target_transpose_current_orientation = exp_target_as_list_given_orientation
                                                    else: _error(f'TRYING TO BUILD DATA_TRANSPOSE BUT NO TARGET OBJECTS GIVEN')

                                                if exp_target_transpose_return_orientation == 'COLUMN':
                                                    EXP_TARGET_TRANSPOSE = transposer(EXP_TARGET_TRANSPOSE)

                                                if exp_target_transpose_return_format == 'ARRAY': pass
                                                elif exp_target_transpose_return_format == 'SPARSE_DICT':
                                                    EXP_TARGET_TRANSPOSE = to_dict(EXP_TARGET_TRANSPOSE)
                                            # END TARGET_TRANSPOSE ######################################################################


                                            # TARGET_AS_LIST ######################################################################
                                            if 'TARGET_AS_LIST' not in RETURN_OBJECTS:
                                                EXP_TARGET_AS_LIST = None
                                                exp_target_as_list_return_format = None
                                                exp_target_as_list_current_format = None
                                                exp_target_as_list_return_orientation = None
                                                exp_target_as_list_current_orientation = None
                                            elif 'TARGET_AS_LIST' in RETURN_OBJECTS:
                                                EXP_TARGET_AS_LIST = BASE_TARGET_OBJECT

                                                # DONT if/elif FORMAT, MUST ALWAYS RETURN AS ARRAY
                                                exp_target_as_list_return_format = 'ARRAY'
                                                exp_target_as_list_current_format = 'ARRAY'

                                                if target_as_list_return_orientation != 'AS_GIVEN':
                                                    exp_target_as_list_return_orientation = target_as_list_return_orientation
                                                    exp_target_as_list_current_orientation = target_as_list_return_orientation
                                                elif target_as_list_return_orientation == 'AS_GIVEN':
                                                    if not GIVEN_TARGET_AS_LIST_OBJECT is None:
                                                        exp_target_as_list_return_orientation = exp_target_as_list_given_orientation
                                                        exp_target_as_list_current_orientation = exp_target_as_list_given_orientation
                                                    elif not GIVEN_TARGET_OBJECT is None:
                                                        exp_target_as_list_return_orientation = exp_target_given_orientation
                                                        exp_target_as_list_current_orientation = exp_target_given_orientation
                                                    elif not GIVEN_TARGET_TRANSPOSE_OBJECT is None:
                                                        exp_target_as_list_return_orientation = exp_target_transpose_given_orientation
                                                        exp_target_as_list_current_orientation = exp_target_transpose_given_orientation
                                                    else: _error(f'TARGET_AS_LIST TO BE RETURNED BUT NO TARGET OBJECTS GIVEN')

                                                if exp_target_as_list_return_orientation == 'ROW':
                                                    EXP_TARGET_AS_LIST = transposer(EXP_TARGET_AS_LIST)

                                                # DONT PUT if blah: to_dict HERE, MUST ALWAYS STAY ARRAY

                                            # END TARGET_AS_LIST ######################################################################


                                            ## END CREATE EXPECTED TARGET RETURN OBJECTS #############################################################################
                                            ########################################################################################################################
                                            ########################################################################################################################

                                            expected_output = (f'Expected output:\n',
                                                    f'exp_this_module = MLObjectOrienter\n',
                                                    f'exp_calling_module = {exp_calling_module}\n',
                                                    f'exp_calling_fxn = {exp_calling_fxn}\n',
                                                    f'single / double = {single_double}\n',
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
                                                    f'exp_xtx_return_format = {exp_xtx_return_format}\n'
                                                    f'exp_xtx_current_format = {exp_xtx_current_format}\n'
                                                    f'exp_xtx_inv_return_format = {exp_xtx_inv_return_format}\n'
                                                    f'exp_xtx_inv_current_format = {exp_xtx_inv_current_format}\n'
                                                    f'target_given_format = {exp_target_given_format}\n'
                                                    f'target_current_format = {exp_target_current_format}\n'
                                                    f'target_return_format = {exp_target_return_format}\n'
                                                    f'target_given_orientation = {exp_target_given_orientation}\n'
                                                    f'target_current_orientation = {exp_target_current_orientation}\n'
                                                    f'target_return_orientation = {exp_target_return_orientation}\n'
                                                    f'target_transpose_given_format = {exp_target_transpose_given_format}\n'
                                                    f'target_transpose_current_format = {exp_target_transpose_current_format}\n'
                                                    f'target_transpose_return_format = {exp_target_transpose_return_format}\n'
                                                    f'target_transpose_given_orientation = {exp_target_transpose_given_orientation}\n'
                                                    f'target_transpose_current_orientation = {exp_target_transpose_current_orientation}\n'
                                                    f'target_transpose_return_orientation = {exp_target_transpose_return_orientation}\n'
                                                    f'target_as_list_given_format = {exp_target_as_list_given_format}\n'
                                                    f'target_as_list_current_format = {exp_target_as_list_current_format}\n'
                                                    f'target_as_list_return_format = {exp_target_as_list_return_format}\n'
                                                    f'target_as_list_given_orientation = {exp_target_as_list_given_orientation}\n'
                                                    f'target_as_list_current_orientation = {exp_target_as_list_current_orientation}\n'
                                                    f'target_as_list_return_orientation= {exp_target_as_list_return_orientation}\n'
                                            )

                                            # print(*expected_output)

                                            Dummy = mloo.MLObjectOrienter(
                                            DATA=None,
                                            data_given_orientation=None,
                                            data_return_orientation=None,
                                            data_return_format=None,
                                            DATA_TRANSPOSE=None,
                                            data_transpose_given_orientation=None,
                                            data_transpose_return_orientation=None,
                                            data_transpose_return_format=None,
                                            XTX=None,
                                            xtx_return_format=None,
                                            XTX_INV=None,
                                            xtx_inv_return_format=None,

                                            ############################################################################################################
                                            TARGET=GIVEN_TARGET_OBJECT,
                                            # ENTER format & orientation for TARGET AS IT WOULD BE IF TARGET IS GIVEN,
                                            # EVEN IF IS NOT GIVEN, IF A TARGET OBJECT IS TO BE RETURNED
                                            # TARGET given_orientation MUST BE ENTERED IF ANY TARGET OBJECT IS TO BE RETURNED
                                            target_given_orientation=target_given_orientation,
                                            target_return_orientation=target_return_orientation,
                                            target_return_format=target_return_format,
                                            target_is_multiclass=is_multiclass,
                                            TARGET_TRANSPOSE=GIVEN_TARGET_TRANSPOSE_OBJECT,
                                            target_transpose_given_orientation=target_transpose_given_orientation,
                                            target_transpose_return_orientation=target_transpose_return_orientation,
                                            target_transpose_return_format=target_transpose_return_format,
                                            TARGET_AS_LIST=GIVEN_TARGET_AS_LIST_OBJECT,
                                            target_as_list_given_orientation=target_as_list_given_orientation,
                                            target_as_list_return_orientation=target_as_list_return_orientation,
                                            ############################################################################################################

                                            RETURN_OBJECTS=RETURN_OBJECTS,
                                            bypass_validation=bypass_validation,
                                            calling_module=calling_module,
                                            calling_fxn=calling_fxn)

                                            act_this_module = 'MLObjectOrienter'
                                            act_calling_module = gmn.get_module_name(str(sys.modules[__name__]))
                                            act_calling_fxn = 'guard_test'

                                            ACT_DATA = Dummy.DATA
                                            act_data_given_orientation = Dummy.data_given_orientation
                                            act_data_given_format = Dummy.data_given_format
                                            act_data_current_orientation = Dummy.data_return_orientation
                                            act_data_current_format = Dummy.data_current_format
                                            act_data_return_orientation = Dummy.data_return_orientation
                                            act_data_return_format = Dummy.data_return_format

                                            ACT_DATA_TRANSPOSE = Dummy.DATA_TRANSPOSE
                                            act_data_transpose_given_orientation = Dummy.data_transpose_given_orientation
                                            act_data_transpose_given_format = Dummy.data_transpose_given_format
                                            act_data_transpose_current_orientation = Dummy.data_transpose_current_orientation
                                            act_data_transpose_current_format = Dummy.data_transpose_current_format
                                            act_data_transpose_return_orientation = Dummy.data_transpose_return_orientation
                                            act_data_transpose_return_format = Dummy.data_transpose_return_format

                                            ACT_XTX = Dummy.XTX
                                            act_xtx_given_format = Dummy.xtx_given_format
                                            act_xtx_return_format = Dummy.xtx_return_format
                                            act_xtx_current_format = Dummy.xtx_current_format

                                            ACT_XTX_INV = Dummy.XTX_INV
                                            act_xtx_inv_given_format = Dummy.xtx_inv_given_format
                                            act_xtx_inv_return_format = Dummy.xtx_inv_return_format
                                            act_xtx_inv_current_format = Dummy.xtx_inv_current_format

                                            ACT_TARGET = Dummy.TARGET
                                            act_is_multiclass = Dummy.target_is_multiclass
                                            act_target_given_format = Dummy.target_given_format
                                            act_target_current_format = Dummy.target_current_format
                                            act_target_return_format = Dummy.target_return_format
                                            act_target_given_orientation = Dummy.target_given_orientation
                                            act_target_current_orientation = Dummy.target_current_orientation
                                            act_target_return_orientation = Dummy.target_return_orientation

                                            ACT_TARGET_TRANSPOSE = Dummy.TARGET_TRANSPOSE
                                            act_target_transpose_given_format = Dummy.target_transpose_given_format
                                            act_target_transpose_current_format = Dummy.target_transpose_current_format
                                            act_target_transpose_return_format = Dummy.target_transpose_return_format
                                            act_target_transpose_given_orientation = Dummy.target_transpose_given_orientation
                                            act_target_transpose_current_orientation = Dummy.target_transpose_current_orientation
                                            act_target_transpose_return_orientation = Dummy.target_transpose_return_orientation

                                            ACT_TARGET_AS_LIST = Dummy.TARGET_AS_LIST
                                            act_target_as_list_given_format = Dummy.target_as_list_given_format
                                            act_target_as_list_current_format = Dummy.target_as_list_current_format
                                            act_target_as_list_return_format = 'ARRAY' if isinstance(ACT_TARGET_AS_LIST, np.ndarray) else 'SPARSE_DICT' if isinstance(ACT_TARGET_AS_LIST, dict) else None
                                            act_target_as_list_given_orientation = Dummy.target_as_list_given_orientation
                                            act_target_as_list_current_orientation = Dummy.target_as_list_current_orientation
                                            act_target_as_list_return_orientation = Dummy.target_as_list_return_orientation



                                            NAMES = [
                                                'calling_module',
                                                'calling_fxn',
                                                'DATA',
                                                'data_given_orientation',
                                                'data_given_format',
                                                'data_current_orientation',
                                                'data_current_format',
                                                'data_return_orientation',
                                                'data_return_format',
                                                'DATA_TRANSPOSE',
                                                'data_transpose_given_orientation',
                                                'data_transpose_given_format',
                                                'data_transpose_current_orientation',
                                                'data_transpose_current_format',
                                                'data_transpose_return_orientation',
                                                'data_transpose_return_format',
                                                'XTX',
                                                'xtx_given_format',
                                                'xtx_return_format',
                                                'xtx_current_format',
                                                'XTX_INV',
                                                'xtx_inv_given_format',
                                                'xtx_inv_return_format',
                                                'xtx_inv_current_format',
                                                'is_multiclass',
                                                'TARGET',
                                                'target_given_format',
                                                'target_current_format',
                                                'target_return_format',
                                                'target_given_orientation',
                                                'target_current_orientation',
                                                'target_return_orientation',
                                                'TARGET_TRANSPOSE',
                                                'target_transpose_given_format',
                                                'target_transpose_current_format',
                                                'target_transpose_return_format',
                                                'target_transpose_given_orientation',
                                                'target_transpose_current_orientation',
                                                'target_transpose_return_orientation',
                                                'TARGET_AS_LIST',
                                                'target_as_list_given_format',
                                                'target_as_list_current_format',
                                                'target_as_list_return_format',
                                                'target_as_list_given_orientation',
                                                'target_as_list_current_orientation',
                                                'target_as_list_return_orientation'
                                            ]

                                            EXPECTED_INPUTS = [
                                                exp_calling_module,
                                                exp_calling_fxn,
                                                EXP_DATA,
                                                exp_data_given_orientation,
                                                exp_data_given_format,
                                                exp_data_current_orientation,
                                                exp_data_current_format,
                                                exp_data_return_orientation,
                                                exp_data_return_format,
                                                EXP_DATA_TRANSPOSE,
                                                exp_data_transpose_given_orientation,
                                                exp_data_transpose_given_format,
                                                exp_data_transpose_current_orientation,
                                                exp_data_transpose_current_format,
                                                exp_data_transpose_return_orientation,
                                                exp_data_transpose_return_format,
                                                EXP_XTX,
                                                exp_xtx_given_format,
                                                exp_xtx_return_format,
                                                exp_xtx_current_format,
                                                EXP_XTX_INV,
                                                exp_xtx_inv_given_format,
                                                exp_xtx_inv_return_format,
                                                exp_xtx_inv_current_format,
                                                act_is_multiclass,
                                                EXP_TARGET,
                                                exp_target_given_format,
                                                exp_target_current_format,
                                                exp_target_return_format,
                                                exp_target_given_orientation,
                                                exp_target_current_orientation,
                                                exp_target_return_orientation,
                                                EXP_TARGET_TRANSPOSE,
                                                exp_target_transpose_given_format,
                                                exp_target_transpose_current_format,
                                                exp_target_transpose_return_format,
                                                exp_target_transpose_given_orientation,
                                                exp_target_transpose_current_orientation,
                                                exp_target_transpose_return_orientation,
                                                EXP_TARGET_AS_LIST,
                                                exp_target_as_list_given_format,
                                                exp_target_as_list_current_format,
                                                exp_target_as_list_return_format,
                                                exp_target_as_list_given_orientation,
                                                exp_target_as_list_current_orientation,
                                                exp_target_as_list_return_orientation,
                                            ]


                                            ACTUAL_OUTPUTS = [
                                                act_calling_module,
                                                act_calling_fxn,
                                                ACT_DATA,
                                                act_data_given_orientation,
                                                act_data_given_format,
                                                act_data_current_orientation,
                                                act_data_current_format,
                                                act_data_return_orientation,
                                                act_data_return_format,
                                                ACT_DATA_TRANSPOSE,
                                                act_data_transpose_given_orientation,
                                                act_data_transpose_given_format,
                                                act_data_transpose_current_orientation,
                                                act_data_transpose_current_format,
                                                act_data_transpose_return_orientation,
                                                act_data_transpose_return_format,
                                                ACT_XTX,
                                                act_xtx_given_format,
                                                act_xtx_return_format,
                                                act_xtx_current_format,
                                                ACT_XTX_INV,
                                                act_xtx_inv_given_format,
                                                act_xtx_inv_return_format,
                                                act_xtx_inv_current_format,
                                                act_is_multiclass,
                                                ACT_TARGET,
                                                act_target_given_format,
                                                act_target_current_format,
                                                act_target_return_format,
                                                act_target_given_orientation,
                                                act_target_current_orientation,
                                                act_target_return_orientation,
                                                ACT_TARGET_TRANSPOSE,
                                                act_target_transpose_given_format,
                                                act_target_transpose_current_format,
                                                act_target_transpose_return_format,
                                                act_target_transpose_given_orientation,
                                                act_target_transpose_current_orientation,
                                                act_target_transpose_return_orientation,
                                                ACT_TARGET_AS_LIST,
                                                act_target_as_list_given_format,
                                                act_target_as_list_current_format,
                                                act_target_as_list_return_format,
                                                act_target_as_list_given_orientation,
                                                act_target_as_list_current_orientation,
                                                act_target_as_list_return_orientation
                                            ]

                                            for description, expected_thing, actual_thing in zip(NAMES, EXPECTED_INPUTS, ACTUAL_OUTPUTS):

                                                try:
                                                    is_equal = np.array_equiv(expected_thing, actual_thing)
                                                    # print(f'\033[91m\n*** TEST EXCEPTED ON np.array_equiv METHOD ***\033[0m\x1B[0m\n')
                                                except: is_equal = None

                                                if is_equal is None:
                                                    try:
                                                        is_equal = expected_thing == actual_thing
                                                    except:
                                                        print(f'{description}:')
                                                        print(f'\n\033[91mEXP_OBJECT = \n{expected_thing}\033[0m\x1B[0m\n')
                                                        print(f'\n\033[91mACT_OBJECT = \n{actual_thing}\033[0m\x1B[0m\n')
                                                        _error(f'\nTEST FAILED "==" METHOD\n')

                                                if not is_equal:
                                                    print(f'*' * 90)
                                                    print(f'Failed on trial {ctr:,} of at most {total_trials:,}')
                                                    print(f'TARGET is {"not given" if GIVEN_TARGET_OBJECT is None else "given"}')
                                                    print(f'TARGET_TRANSPOSE is {"not given" if GIVEN_TARGET_TRANSPOSE_OBJECT is None else "given"}')
                                                    print(f'TARGET_AS_LIST is {"not given" if GIVEN_TARGET_AS_LIST_OBJECT is None else "given"}')
                                                    print(f'RETURN_OBJECTS = ')
                                                    print(RETURN_OBJECTS)
                                                    print()
                                                    print(*expected_output)
                                                    print()
                                                    print(f'\n\033[91mEXP_OBJECT = \n{expected_thing}\033[0m\x1B[0m\n')
                                                    print(f'\n\033[91mACT_OBJECT = \n{actual_thing}\033[0m\x1B[0m\n')
                                                    _error(
                                                        f'\n{description} FAILED EQUALITY TEST, \nexpected = \n{expected_thing}\n'
                                                        f'actual = \n{actual_thing}\n')
                                                else:
                                                    pass  # print(f'\033[92m     *** {description} PASSED ***\033[0m\x1B[0m')


print(f'\n\033[92m*** TEST DONE. ALL PASSED. ***\033[0m\x1B[0m\n')






















