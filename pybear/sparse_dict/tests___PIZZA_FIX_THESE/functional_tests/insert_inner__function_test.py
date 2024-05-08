import sys, inspect, warnings, time
import numpy as np
import sparse_dict as sd
from MLObjects.TestObjectCreators import test_header as th


# 3/26/23 PIZZA VERIFIED MODULES AND TEST CODE ARE GOOD.

# 3/25/23 TEST sd.core_insert_inner, sd.insert_inner, sd.append_inner FOR FUNCTIONALITY

# CAN ONLY PROVIDE DICT AS INPUT, INSERT OBJECTS CAN BE NP OR SD


def test_fail(test_name, oper_desc, GIVEN_OBJECT, EXP_OBJECT, ACT_OBJECT):
    print('\033[91m')
    print(f'\n*** {test_name} EPIC FAIL ***\n', )
    print(f'\033[92mOPERATION DESCRIPTION:\n', oper_desc)
    print(f'\n\033[92mGIVEN OBJECT:\n', [f'{_}: {GIVEN_OBJECT[_]}' for _ in range(len(GIVEN_OBJECT))])
    print(f'\n\033[91mEXPECTED OBJECT:\n', EXP_OBJECT)
    print(f'\nACTUAL OBJECT:\n', ACT_OBJECT)
    print('\033[0m')
    raise Exception(f'*** EPIC FAIL 😂😂😂 ***')


NP_BASE_INPUT_OBJ = np.fromiter(range(25), dtype=np.int8).reshape((5,-1))
GIVEN_INPUT_DICT = sd.zip_list_as_py_int(NP_BASE_INPUT_OBJ)

# CONVERT NP_BASE_INPUT_OBJ TO SD AFTER BUILDING NP PRECURSOR OF EXP_DICT DURING RUN

MASTER_OBJ_HEADER = ['build_in_process', None]
MASTER_HEADER_AXIS = [0, 1]
MASTER_INSERT_IDX = [0,3,5]
MASTER_INSERT_HEADER = ['build_in_process', None]
MASTER_INSERT_SIZE = [1,2]
MASTER_INSERT_FORMAT = ['ARRAY','SPARSE_DICT']
MASTER_INSERT_ORIENTATION = ['ROW','COLUMN']
MASTER_INSERT_SINGLE_DOUBLE = ['SINGLE','DOUBLE']

########################################################################################################################
# TEST core_insert_inner ###############################################################################################
print(f'\n\033[92m*** STARTING core_insert_inner() TEST ***\033[0m')

total_trials = np.product(list(map(len, (MASTER_INSERT_IDX, MASTER_INSERT_SIZE, MASTER_INSERT_FORMAT,
                                         MASTER_INSERT_ORIENTATION, MASTER_INSERT_SINGLE_DOUBLE))))

ctr = 0
for ins_idx in MASTER_INSERT_IDX:
    for ins_size in MASTER_INSERT_SIZE:
        for ins_format in MASTER_INSERT_FORMAT:
            for ins_orientation in MASTER_INSERT_ORIENTATION:
                for single_double in MASTER_INSERT_SINGLE_DOUBLE:
                    ctr += 1
                    print(f'\033[92mRunning trial {ctr} of {total_trials}...\033[0m')

                    test_name = 'core_insert_inner'

                    if ins_size == 2: single_double = 'DOUBLE'
                    if ins_orientation == 'ROW': single_double = 'DOUBLE'

                    oper_desc = f'\033[92mInsert a {single_double}{" ( [[]] )" if single_double=="DOUBLE" else " ( [] )" if single_double=="SINGLE" else ""} ' \
                                f'{ins_format} of 5 rows by {ins_size} columns oriented as {ins_orientation} into the {ins_idx} inner ' \
                                f'index position of a 5 x 5 SPARSE_DICT.\033[0m'

                    # BUILD GIVEN INSERT #############################################################################
                    if ins_size == 1: BASE_INSERT_OBJ = np.fromiter(range(95,100), dtype=np.int8).reshape((1,-1))
                    elif ins_size == 2: BASE_INSERT_OBJ = np.fromiter(range(90,100), dtype=np.int8).reshape((2,-1))

                    if ins_orientation == 'ROW': BASE_INSERT_OBJ = BASE_INSERT_OBJ.transpose()

                    if ins_format == 'SPARSE_DICT': BASE_INSERT_OBJ = sd.zip_list_as_py_int(BASE_INSERT_OBJ)

                    if single_double == 'SINGLE' and ins_size==1 and ins_orientation=='COLUMN':
                        BASE_INSERT_OBJ = BASE_INSERT_OBJ[0]

                    GIVEN_INS_OBJ = BASE_INSERT_OBJ
                    # END BUILD GIVEN INSERT #############################################################################

                    # BUILD EXP_DICT #####################################################################################

                    EXP_DICT = NP_BASE_INPUT_OBJ.copy()
                    if ins_size==1: DUM_INSERT = np.fromiter(range(95,100),dtype=np.int8).reshape((1,-1))[0]
                    elif ins_size==2: DUM_INSERT = np.fromiter(range(90,100),dtype=np.int8).reshape((2,-1))
                    EXP_DICT = np.insert(EXP_DICT, ins_idx, DUM_INSERT, axis=1); del DUM_INSERT
                    EXP_DICT = sd.zip_list_as_py_int(EXP_DICT)
                    # END BUILD EXP_DICT #################################################################################

                    # HEADER IS NOT A FEATURE OF core_insert_inner
                    ACT_DICT = sd.core_insert_inner(GIVEN_INPUT_DICT, ins_idx, GIVEN_INS_OBJ)

                    if not sd.core_sparse_equiv(EXP_DICT, ACT_DICT):
                        test_fail('core_insert_inner', oper_desc, GIVEN_INPUT_DICT, EXP_DICT, ACT_DICT)

print(f'\n\033[92m*** core_insert_inner TESTS COMPLETE. ALL TESTS PASSED. ***\033[0m')
# END TEST core_insert_inner ###########################################################################################
########################################################################################################################

print(f'\n' + f'X'*100)

########################################################################################################################
# TEST insert_inner ####################################################################################################
print(f'\n\033[92m*** STARTING insert_inner() TEST ***\033[0m')

MASTER_HEADER_SINGLE_DOUBLE = ['SINGLE', 'DOUBLE']

total_trials = np.product(list(map(len, (MASTER_OBJ_HEADER, MASTER_HEADER_AXIS, MASTER_INSERT_IDX, MASTER_INSERT_HEADER,
     MASTER_HEADER_SINGLE_DOUBLE, MASTER_INSERT_SIZE, MASTER_INSERT_FORMAT, MASTER_INSERT_ORIENTATION, MASTER_INSERT_SINGLE_DOUBLE))))

ctr = 0
for obj_header_desc in MASTER_OBJ_HEADER:
    for header_axis in MASTER_HEADER_AXIS:
        for ins_idx in MASTER_INSERT_IDX:
            for ins_header_desc in MASTER_INSERT_HEADER:
                for hdr_single_double in MASTER_HEADER_SINGLE_DOUBLE:
                    for ins_size in MASTER_INSERT_SIZE:
                        for ins_format in MASTER_INSERT_FORMAT:
                            for ins_orientation in MASTER_INSERT_ORIENTATION:
                                for obj_single_double in MASTER_INSERT_SINGLE_DOUBLE:
                                    ctr += 1
                                    print(f'\033[92mRunning trial {ctr} of {total_trials}...\033[0m')

                                    test_name = 'insert_inner'

                                    if ins_size == 2: obj_single_double = 'DOUBLE'
                                    if ins_orientation == 'ROW': obj_single_double = 'DOUBLE'

                                    oper_desc = f'\033[92mInsert a {obj_single_double}{" ( [[]] )" if obj_single_double == "DOUBLE" else " ( [] )" if obj_single_double == "SINGLE" else ""} ' \
                                                f'{ins_format} of 5 rows by {ins_size} columns oriented as {ins_orientation} into the {ins_idx} inner ' \
                                                f'index position of a 5 x 5 SPARSE_DICT.' \
                                                f'\nINPUT DICT header is{" not" if obj_header_desc is None else " "}given{"." if obj_header_desc is None else " along the "}' \
                                                f'{"" if obj_header_desc is None else header_axis} axis (insertion only happens along the 1 axis). ' \
                                                f'INSERT header is{" not" if ins_header_desc is None else " "}given.\033[0m\n'

                                    # BUILD GIVEN HEADER #############################################################################
                                    if obj_header_desc == 'build_in_process':
                                        GIVEN_OBJ_HEADER = th.test_header(5)
                                        if hdr_single_double == 'DOUBLE': pass
                                        elif hdr_single_double == 'SINGLE': GIVEN_OBJ_HEADER = GIVEN_OBJ_HEADER[0]
                                    elif obj_header_desc == None: GIVEN_OBJ_HEADER = None
                                    # END BUILD GIVEN HEADER ##########################################################################

                                    # BUILD GIVEN INSERT #############################################################################
                                    if ins_size == 1: BASE_INSERT_OBJ = np.fromiter(range(95, 100), dtype=np.int8).reshape((1, -1))
                                    elif ins_size == 2: BASE_INSERT_OBJ = np.fromiter(range(90, 100), dtype=np.int8).reshape((2, -1))

                                    if ins_orientation == 'ROW': BASE_INSERT_OBJ = BASE_INSERT_OBJ.transpose()

                                    if ins_format == 'SPARSE_DICT': BASE_INSERT_OBJ = sd.zip_list_as_py_int(BASE_INSERT_OBJ)

                                    if obj_single_double == 'SINGLE' and ins_size == 1 and ins_orientation == 'COLUMN':
                                        BASE_INSERT_OBJ = BASE_INSERT_OBJ[0]

                                    GIVEN_INS_OBJ = BASE_INSERT_OBJ
                                    # END BUILD GIVEN INSERT #############################################################################

                                    # BUILD GIVEN INSERT HEADER ###########################################################################
                                    if ins_header_desc == 'build_in_process':
                                        if ins_size==1:
                                            GIVEN_INS_HEADER = np.array(['INS_TEST_1'],dtype='<U15').reshape((1, -1))
                                            if hdr_single_double == 'DOUBLE': pass
                                            elif hdr_single_double == 'SINGLE': GIVEN_INS_HEADER = GIVEN_INS_HEADER[0]
                                        elif ins_size == 2:   # MUST BE DOUBLE
                                            GIVEN_INS_HEADER = np.array(['INS_TEST_1','INS_TEST_2'], dtype='<U15').reshape((1, -1))
                                    elif ins_header_desc is None:
                                        GIVEN_INS_HEADER = None
                                    # END BUILD GIVEN INSERT HEADER #######################################################################

                                    # BUILD EXP_DICT #####################################################################################
                                    EXP_DICT = NP_BASE_INPUT_OBJ.copy()
                                    if ins_size == 1: DUM_INSERT = np.fromiter(range(95, 100), dtype=np.int8).reshape((1, -1))[0]
                                    elif ins_size == 2: DUM_INSERT = np.fromiter(range(90, 100), dtype=np.int8).reshape((2, -1))

                                    EXP_DICT = np.insert(EXP_DICT, ins_idx, DUM_INSERT, axis=1)
                                    del DUM_INSERT
                                    EXP_DICT = sd.zip_list_as_py_int(EXP_DICT)
                                    # END BUILD EXP_DICT #################################################################################

                                    # BUILD EXP HEADER ###################################################################################
                                    # IF (INS AND DICT1 HEADERS GIVEN, OR ONE OR THE OTHER GIVEN) THEN A HEADER SHOULD BE RETURNED BY insert_inner OTHERWISE RETURNS None
                                    if obj_header_desc is None and ins_header_desc is None:
                                        EXP_HEADER = None
                                    else:
                                        # THESE ARE THE DUMMY OUTPUT FOR NOT-GIVEN HEADERS FROM sparse_dict.header_handle()
                                        DUM_OBJ_HEADER = np.fromiter((f'DICT1_COL_{idx+1}' for idx in range(5)), dtype='<U20').reshape((1,-1))
                                        DUM_INS_HEADER = np.fromiter((f'INS_COL_{idx+1}' for idx in range(ins_size)), dtype='<U20').reshape((1,-1))

                                        # ALL INSERTIONS ONLY CHANGE AXIS 1. SO IF HEADER IS AXIS 0 THERE IS NO CHANGE TO GIVEN_OBJ_HEADER
                                        if header_axis==0:
                                            if not obj_header_desc is None: EXP_HEADER = GIVEN_OBJ_HEADER
                                            elif obj_header_desc is None: EXP_HEADER = DUM_OBJ_HEADER
                                        elif header_axis==1:
                                            if not obj_header_desc is None and not ins_header_desc is None:
                                                EXP_HEADER = np.hstack((GIVEN_OBJ_HEADER.reshape((1,-1))[...,:ins_idx], GIVEN_INS_HEADER.reshape((1,-1)), GIVEN_OBJ_HEADER.reshape((1,-1))[...,ins_idx:]))
                                            elif not obj_header_desc is None and ins_header_desc is None:
                                                EXP_HEADER = np.hstack((GIVEN_OBJ_HEADER.reshape((1,-1))[...,:ins_idx], DUM_INS_HEADER.reshape((1,-1)), GIVEN_OBJ_HEADER.reshape((1,-1))[...,ins_idx:]))
                                            elif obj_header_desc is None and not ins_header_desc is None:
                                                EXP_HEADER = np.hstack((DUM_OBJ_HEADER[..., :ins_idx], GIVEN_INS_HEADER.reshape((1,-1)), DUM_OBJ_HEADER[..., ins_idx:]))

                                        del DUM_OBJ_HEADER, DUM_INS_HEADER

                                    # END BUILD EXP HEADER ###################################################################################

                                    if ins_header_desc is None and obj_header_desc is None:

                                        ACT_DICT = sd.insert_inner(GIVEN_INPUT_DICT, ins_idx, GIVEN_INS_OBJ,
                                           DICT_HEADER1=GIVEN_OBJ_HEADER, INSERT_HEADER=GIVEN_INS_HEADER, header_axis=header_axis, fxn='insert_inner_test')

                                        if not sd.core_sparse_equiv(EXP_DICT, ACT_DICT):
                                            test_fail('insert_inner', oper_desc, GIVEN_INPUT_DICT, EXP_DICT, ACT_DICT)

                                    else: # elif not ins_header_desc is None and not obj_header_desc is None:
                                        ACT_DICT, ACT_DICT_HEADER = sd.insert_inner(GIVEN_INPUT_DICT, ins_idx, GIVEN_INS_OBJ,
                                            DICT_HEADER1=GIVEN_OBJ_HEADER, INSERT_HEADER=GIVEN_INS_HEADER, header_axis=header_axis, fxn='insert_inner_test')

                                        if not sd.core_sparse_equiv(EXP_DICT, ACT_DICT):
                                            test_fail('insert_inner', oper_desc, GIVEN_INPUT_DICT, EXP_DICT, ACT_DICT)

                                        if not np.array_equiv(EXP_HEADER, ACT_DICT_HEADER):
                                            test_fail('insert_inner', oper_desc, GIVEN_OBJ_HEADER, EXP_HEADER, ACT_DICT_HEADER)

print(f'\n\033[92m*** insert_inner() TESTS COMPLETE. ALL TESTS PASSED. ***\033[0m')
# END TEST insert_inner ################################################################################################
########################################################################################################################

print(f'\n' + f'X'*100)

########################################################################################################################
# TEST append_inner ####################################################################################################
print(f'\n\033[92m*** STARTING append_inner() TEST ***\033[0m')

MASTER_HEADER_SINGLE_DOUBLE = ['SINGLE', 'DOUBLE']

total_trials = np.product(list(map(len, (MASTER_OBJ_HEADER, MASTER_HEADER_AXIS, MASTER_INSERT_HEADER,
                                         MASTER_HEADER_SINGLE_DOUBLE, MASTER_INSERT_SIZE, MASTER_INSERT_FORMAT,
                                         MASTER_INSERT_ORIENTATION, MASTER_INSERT_SINGLE_DOUBLE))))

ctr = 0
for obj_header_desc in MASTER_OBJ_HEADER:
    for header_axis in MASTER_HEADER_AXIS:
        for ins_header_desc in MASTER_INSERT_HEADER:
            for hdr_single_double in MASTER_HEADER_SINGLE_DOUBLE:
                for ins_size in MASTER_INSERT_SIZE:
                    for ins_format in MASTER_INSERT_FORMAT:
                        for ins_orientation in MASTER_INSERT_ORIENTATION:
                            for obj_single_double in MASTER_INSERT_SINGLE_DOUBLE:
                                ctr += 1
                                print(f'\033[92mRunning trial {ctr} of {total_trials}...\033[0m')

                                test_name = 'append_inner'

                                if ins_size == 2: obj_single_double = 'DOUBLE'
                                if ins_orientation == 'ROW': obj_single_double = 'DOUBLE'

                                oper_desc = f'\033[92mAppend a {obj_single_double}{" ( [[]] )" if obj_single_double == "DOUBLE" else " ( [] )" if obj_single_double == "SINGLE" else ""} ' \
                                            f'{ins_format} of 5 rows by {ins_size} columns oriented as {ins_orientation} into the last inner ' \
                                            f'index position of a 5 x 5 SPARSE_DICT.' \
                                            f'\nINPUT DICT header is{" not" if obj_header_desc is None else " "}given{"." if obj_header_desc is None else " along the "}' \
                                            f'{"" if obj_header_desc is None else header_axis} axis (insertion only happens along the 1 axis). ' \
                                            f'INSERT header is{" not" if ins_header_desc is None else " "}given.\033[0m\n'

                                # BUILD GIVEN HEADER #############################################################################
                                if obj_header_desc == 'build_in_process':
                                    GIVEN_OBJ_HEADER = th.test_header(5)
                                    if hdr_single_double == 'DOUBLE': pass
                                    elif hdr_single_double == 'SINGLE': GIVEN_OBJ_HEADER = GIVEN_OBJ_HEADER[0]
                                elif obj_header_desc == None: GIVEN_OBJ_HEADER = None
                                # END BUILD GIVEN HEADER ##########################################################################

                                # BUILD GIVEN INSERT #############################################################################
                                if ins_size == 1: BASE_INSERT_OBJ = np.fromiter(range(95, 100), dtype=np.int8).reshape((1, -1))
                                elif ins_size == 2: BASE_INSERT_OBJ = np.fromiter(range(90, 100), dtype=np.int8).reshape((2, -1))

                                if ins_orientation == 'ROW': BASE_INSERT_OBJ = BASE_INSERT_OBJ.transpose()

                                if ins_format == 'SPARSE_DICT': BASE_INSERT_OBJ = sd.zip_list_as_py_int(BASE_INSERT_OBJ)

                                if obj_single_double == 'SINGLE' and ins_size == 1 and ins_orientation == 'COLUMN':
                                    BASE_INSERT_OBJ = BASE_INSERT_OBJ[0]

                                GIVEN_INS_OBJ = BASE_INSERT_OBJ
                                # END BUILD GIVEN INSERT #############################################################################

                                # BUILD GIVEN INSERT HEADER ###########################################################################
                                if ins_header_desc == 'build_in_process':
                                    if ins_size == 1:
                                        GIVEN_INS_HEADER = np.array(['INS_TEST_1'], dtype='<U15').reshape((1, -1))
                                        if hdr_single_double == 'DOUBLE': pass
                                        elif hdr_single_double == 'SINGLE': GIVEN_INS_HEADER = GIVEN_INS_HEADER[0]
                                    elif ins_size == 2:  # MUST BE DOUBLE
                                        GIVEN_INS_HEADER = np.array(['INS_TEST_1', 'INS_TEST_2'], dtype='<U15').reshape((1, -1))
                                elif ins_header_desc is None: GIVEN_INS_HEADER = None
                                # END BUILD GIVEN INSERT HEADER #######################################################################

                                # BUILD EXP_DICT #####################################################################################
                                EXP_DICT = NP_BASE_INPUT_OBJ.copy()
                                if ins_size == 1: DUM_INSERT = np.fromiter(range(95, 100), dtype=np.int8).reshape((1, -1))[0]
                                elif ins_size == 2: DUM_INSERT = np.fromiter(range(90, 100), dtype=np.int8).reshape((2, -1))

                                EXP_DICT = np.insert(EXP_DICT, len(EXP_DICT), DUM_INSERT, axis=1)
                                del DUM_INSERT
                                EXP_DICT = sd.zip_list_as_py_int(EXP_DICT)
                                # END BUILD EXP_DICT #################################################################################

                                # BUILD EXP HEADER ###################################################################################
                                # IF (INS AND DICT1 HEADERS GIVEN, OR ONE OR THE OTHER GIVEN) THEN A HEADER SHOULD BE RETURNED BY insert_inner OTHERWISE RETURNS None
                                if obj_header_desc is None and ins_header_desc is None:
                                    EXP_HEADER = None
                                else:
                                    # THESE ARE THE DUMMY OUTPUT FOR NOT-GIVEN HEADERS FROM sparse_dict.header_handle()
                                    DUM_OBJ_HEADER = np.fromiter((f'DICT1_COL_{idx + 1}' for idx in range(5)), dtype='<U20').reshape((1, -1))
                                    DUM_INS_HEADER = np.fromiter((f'INS_COL_{idx + 1}' for idx in range(ins_size)), dtype='<U20').reshape((1, -1))

                                    # ALL INSERTIONS ONLY CHANGE AXIS 1. SO IF HEADER IS AXIS 0 THERE IS NO CHANGE TO GIVEN_OBJ_HEADER
                                    if header_axis == 0:
                                        if not obj_header_desc is None: EXP_HEADER = GIVEN_OBJ_HEADER
                                        elif obj_header_desc is None: EXP_HEADER = DUM_OBJ_HEADER
                                    elif header_axis == 1:
                                        if not obj_header_desc is None and not ins_header_desc is None:
                                            EXP_HEADER = np.hstack((GIVEN_OBJ_HEADER.reshape((1, -1)), GIVEN_INS_HEADER.reshape((1, -1))))
                                        elif not obj_header_desc is None and ins_header_desc is None:
                                            EXP_HEADER = np.hstack((GIVEN_OBJ_HEADER.reshape((1, -1)), DUM_INS_HEADER.reshape((1, -1))))
                                        elif obj_header_desc is None and not ins_header_desc is None:
                                            EXP_HEADER = np.hstack((DUM_OBJ_HEADER, GIVEN_INS_HEADER.reshape((1, -1))))

                                    del DUM_OBJ_HEADER, DUM_INS_HEADER

                                # END BUILD EXP HEADER ###################################################################################

                                if ins_header_desc is None and obj_header_desc is None:

                                    ACT_DICT = sd.append_inner(GIVEN_INPUT_DICT,
                                                               GIVEN_INS_OBJ,
                                                               DICT_HEADER1=GIVEN_OBJ_HEADER,
                                                               INSERT_HEADER=GIVEN_INS_HEADER,
                                                               header_axis=header_axis,
                                                               fxn='append_inner_test')

                                    if not sd.core_sparse_equiv(EXP_DICT, ACT_DICT):
                                        test_fail('append_inner', oper_desc, GIVEN_INPUT_DICT, EXP_DICT, ACT_DICT)

                                else:  # elif not ins_header_desc is None and not obj_header_desc is None:
                                    ACT_DICT, ACT_DICT_HEADER = sd.append_inner(GIVEN_INPUT_DICT,
                                                                                GIVEN_INS_OBJ,
                                                                                DICT_HEADER1=GIVEN_OBJ_HEADER,
                                                                                INSERT_HEADER=GIVEN_INS_HEADER,
                                                                                header_axis=header_axis,
                                                                                fxn='append_inner_test')

                                    if not sd.core_sparse_equiv(EXP_DICT, ACT_DICT):
                                        test_fail('append_inner', oper_desc, GIVEN_INPUT_DICT, EXP_DICT, ACT_DICT)

                                    if not np.array_equiv(EXP_HEADER, ACT_DICT_HEADER):
                                        test_fail('append_inner', oper_desc, GIVEN_OBJ_HEADER, EXP_HEADER,
                                                  ACT_DICT_HEADER)

print(f'\n\033[92m*** append_inner() TESTS COMPLETE. ALL TESTS PASSED. ***\033[0m')
# END TEST append_inner ################################################################################################
########################################################################################################################



print(f'\n\033[92m*** TESTS COMPLETE. ALL TESTS PASSED. ***\033[0m')






