import sys, inspect, time
from general_sound import winlinsound as wls
from copy import deepcopy
import numpy as np, sparse_dict as sd
from debug import get_module_name as gmn
from data_validation import arg_kwarg_validater as akv
from ML_PACKAGE._data_validation import list_dict_validater as ldv
from general_data_ops import get_shape as gs
from MLObjects import MLRowColumnOperations as mlrco


# THIS MODULE CAN ONLY BE PASSED ONE OBJECT, AND CAN ONLY RETURN (TRAIN_OBJ1, TEST_OBJ1) OR (TRAIN_OBJ1, DEV_OBJ1, TEST_OBJ1)


# NOT ADVISED TO USE THIS AS IS, USE general_data_ops.TrainDevTestSplit
# HAVING THIS SEPARATE HAS BEEN CONVENIENT FOR TEST 3/27/23
# MASK MUST BE GENERATED EXTERNALLY AND PASSED AS LIST-TYPE OF 1 (FOR RETURN 2) OR 2 COLUMNS (FOR RETURN 3)
# THE CHOPPED PASSED OBJECT IS ALWAYS RETURNED FIRST, THEN THE ROWS INDICATED BY MASK'S RESPECTIVE COLUMNS SEQUENTIALLY.
# MASK ENTRIES MUST BE BIN OR BOOL, True INDICATING KEEP THAT ROW FOR THAT OBJECT.  MAX ONE TRUE PER ROW.


def train_dev_test_split_core(OBJECT, given_orientation, MASK, bypass_validation=None):


    def _exception(words): raise Exception(f'{this_module} >>> {words}')


    this_module = gmn.get_module_name(str(sys.modules[__name__]))
    fxn = inspect.stack()[0][3]

    bypass_validation = akv.arg_kwarg_validater(bypass_validation, 'bypass_validation', [True,False,None],
                                                this_module, fxn, return_if_none=False)

    is_list = isinstance(OBJECT, (list, tuple, np.ndarray))
    is_dict = isinstance(OBJECT, dict)
    is_row = given_orientation=='ROW'
    is_col = given_orientation=='COLUMN'

    # ldv FOR MASK IRREGARDLESS OF validation, MUST GET IT TO [[]] IF GIVEN AS SINGLE
    mask_format, MASK = ldv.list_dict_validater(MASK, 'MASK')
    if not mask_format == 'ARRAY': _exception(f'MASK must be passed AS ARRAY')
    del mask_format
    MASK = MASK.astype(bool)

    if not bypass_validation:

        if not is_list and not is_dict: _exception(f'OBJECT must be a dict or a list-type.')
        if not is_row and not is_col: _exception(f'given_orientation "{given_orientation}" must be "ROW" or "COLUMN".')

        # MASK SHOULD BE ORIENTED AS [[]=COLUMNS] BUT CHECK IF IT IS [[]=ROWS] AND WOULD BE CORRECT IF TRANSPOSED
        mask_rows, mask_columns = gs.get_shape('MASK', MASK, 'COLUMN')
        obj_rows = gs.get_shape('OBJECT', OBJECT, given_orientation)[0]
        if not obj_rows==mask_rows:
            if mask_columns==obj_rows: MASK = MASK.transpose()
            else:  # elif NEITHER MASK ROWS NOR COLUMNS MATCH AGAINST OBJECTS ROWS
                _exception(f'Number of rows in OBJECT and MASK are not equal.')
        del mask_rows, obj_rows

        if not mask_columns in [1,2]:
            _exception(f'Number of columns in MASK must be 1 (for double return) or 2 (for triple return).')

        del mask_columns

        # MASK MUST SUM ACROSS ROWS astype(int8) TO ZERO OR ONE FOR EACH ROW.  MUST HAVE ZEROS AND ONES.
        # ALL ZEROS WOULD NOT MOVE ANY COLUMNS (DEV/TEST STAYS EMPTY) ALL ONES WOULD EMPTY TRAIN.
        ROW_SUMS = np.sum(MASK.astype(np.int8), axis=0).tolist()
        if not (max(ROW_SUMS)==1 and min(ROW_SUMS)==0):
            _exception(f'MASK ROWS MUST SUM TO ZEROS AND ONES.  CANNOT BE ALL ZEROS OR ALL ONES.')
        del ROW_SUMS

    if is_dict:
        print(f'Working in sparse dict mode.  Patience...')
        # MUST TRANSFORM MASK INTO LIST(S) OF IDXS FOR sd select AND delete FUNCTIONS
        MASK_FOR_DICT = np.empty(len(MASK), dtype=object)
        for col_idx in range(len(MASK)):
            MASK_FOR_DICT[col_idx] = np.nonzero(MASK[col_idx])[-1]

    # NUMBER OF COLUMNS IN MASK DETERMINES NUMBER OF COLUMNS TO RETURN
    if len(MASK)==1:
        MASK = np.nonzero(MASK[0])[-1].reshape((1,-1))[0]

        WorkingObject = mlrco.MLRowColumnOperations(OBJECT, given_orientation, name='OBJECT',bypass_validation=bypass_validation)

        # SELECT EXAMPLES FOR DEV_OR_TEST_OBJECTS CALLED OUT IN MASK FROM OBJECT
        DEV_OR_TEST_OBJECT = WorkingObject.return_rows(MASK, return_orientation='AS_GIVEN', return_format='AS_GIVEN')

        # DELETE EXAMPLES FROM OBJECT
        OBJECT = WorkingObject.delete_rows(MASK)

        del WorkingObject

        SPLIT_TUPLE = (OBJECT, DEV_OR_TEST_OBJECT)

    elif len(MASK)==2:
        # USE 0 SLOT TO BUILD DEV & 1 SLOT TO BUILD TEST
        WorkingObject = mlrco.MLRowColumnOperations(OBJECT, given_orientation, name='OBJECT', bypass_validation=bypass_validation)

        for obj_idx in range(0,2):
            # SELECT EXAMPLES FOR DEV_OR_TEST_OBJECTS CALLED OUT IN MASK FROM OBJECT
            WIP_MASK = np.nonzero(MASK[obj_idx])[-1].reshape((1,-1))[0]

            DEV_OR_TEST_OBJECT = WorkingObject.return_rows(WIP_MASK, return_orientation='AS_GIVEN', return_format='AS_GIVEN')

            del WIP_MASK

            if obj_idx==0: DEV_OBJECT = deepcopy(DEV_OR_TEST_OBJECT) if is_dict else DEV_OR_TEST_OBJECT.copy()
            elif obj_idx==1: TEST_OBJECT = deepcopy(DEV_OR_TEST_OBJECT) if is_dict else DEV_OR_TEST_OBJECT.copy()

        # DELETE EXAMPLES FROM OBJECT, MUST SUM THE TWO MASK COLUMNS
        MASK = np.nonzero(np.sum(MASK.astype(np.int8), axis=0).astype(bool))[-1].reshape((1,-1))[0]
        OBJECT = WorkingObject.delete_rows(MASK)

        SPLIT_TUPLE = (OBJECT, DEV_OBJECT, TEST_OBJECT)

    if is_dict:
        del MASK_FOR_DICT
        print(f'Done working in sparse dict mode.')

    del MASK, _exception

    return SPLIT_TUPLE








if __name__ == '__main__':
    # TEST MODULE

    # FUNCTIONAL MODULE & TEST MODULE VERIFIED GOOD 3/26/2023.

    # TESTS THE ACCURACY OF OUTPUT OBJECTS FOR A GIVEN MASK

    BASE_OBJECT = np.fromiter(range(36), dtype=np.int8).reshape((6,6))
    MASTER_OBJECTS = ['np_square','sd_square']
    MASTER_G_O = ['ROW','COLUMN']
    MASTER_MASKS = ['2_object_edges','2_object_middle','3_object_edges','3_object_middle']
    MASTER_B_V = [True, False, None]

    total_trials = np.product(list(map(len, (MASTER_OBJECTS, MASTER_G_O, MASTER_MASKS, MASTER_B_V))))

    def test_except(words):
        raise Exception(f'\033[91m *** TEST EXCEPTION >>> {words} *** \033[0m')

    ctr = 0
    for object_format in MASTER_OBJECTS:
        for given_orientation in MASTER_G_O:
            for mask_format in MASTER_MASKS:
                for bypass_validation in MASTER_B_V:
                    ctr += 1
                    print(f'\033[92mRunning trial {ctr} of {total_trials}\033[0m')
                    if object_format == 'np_square' and given_orientation == 'COLUMN':
                        pass
                    elif object_format == 'np_square' and given_orientation == 'ROW':
                        GIVEN_OBJECT = BASE_OBJECT.transpose()
                    elif object_format == 'sd_square' and given_orientation == 'COLUMN':
                        GIVEN_OBJECT = sd.zip_list_as_py_int(BASE_OBJECT)
                    elif object_format == 'sd_square' and given_orientation == 'ROW':
                        GIVEN_OBJECT = sd.zip_list_as_py_int(BASE_OBJECT.transpose())

                    if mask_format == '2_object_edges':
                        GIVEN_MASK = [[False, True, True, True, True, False]]
                    elif mask_format == '2_object_middle':
                        GIVEN_MASK = [[True, False, False, False, False, True]]
                    elif mask_format == '3_object_edges':
                        GIVEN_MASK = [[False, True, False, False, True, False],
                                      [True, False, False, False, False, True]]
                    elif mask_format == '3_object_middle':
                        GIVEN_MASK = [[False, True, False, False, True, False],
                                      [False, False, True, True, False, False]]

                    ACTUAL = train_dev_test_split_core(GIVEN_OBJECT, given_orientation, GIVEN_MASK, bypass_validation=None)

                    obj_desc = f'\033[92m' \
                               f'\n' \
                               f'Expected objects are {"TRAIN-TEST" if mask_format[0]=="2" else "TRAIN-DEV-TEST"} ' \
                               f'{"sparse dict" if object_format[:2]=="sd" else "array"}s oriented as {given_orientation}. ' \
                               f'bypass_validation = {bypass_validation}' \
                               f'\n' \
                               f'\033[0m'

                    print(obj_desc)

                    # TEST NUMBER OF OBJECTS RETURNED
                    if mask_format[0]=='2' and len(ACTUAL)!=2:
                        test_except(f'EXPECTED 2 OBJECTS RETURNED, GOT {len(ACTUAL)}')
                    elif mask_format[0]=='3' and len(ACTUAL)!=3:
                        test_except(f'EXPECTED 3 OBJECTS RETURNED, GOT {len(ACTUAL)}')

                    if object_format == 'np_square':
                        if given_orientation == 'ROW':
                            if mask_format[0] == '2': EXPECTED = (np.delete(GIVEN_OBJECT, GIVEN_MASK[0], axis=0),
                                                                  GIVEN_OBJECT[GIVEN_MASK[0], ...])
                            elif mask_format[0] == '3': EXPECTED = (np.delete(GIVEN_OBJECT, np.sum(GIVEN_MASK, axis=0).astype(bool), axis=0),
                                                                    GIVEN_OBJECT[GIVEN_MASK[0], ...], GIVEN_OBJECT[GIVEN_MASK[1], ...])
                        elif given_orientation == 'COLUMN':
                            if mask_format[0] == '2': EXPECTED = (np.delete(GIVEN_OBJECT, GIVEN_MASK[0], axis=1),
                                                                  GIVEN_OBJECT[..., GIVEN_MASK[0]])
                            elif mask_format[0] == '3': EXPECTED = (np.delete(GIVEN_OBJECT, np.sum(GIVEN_MASK, axis=0).astype(bool), axis=1),
                                                                    GIVEN_OBJECT[..., GIVEN_MASK[0]], GIVEN_OBJECT[..., GIVEN_MASK[1]])

                    elif object_format == 'sd_square':
                        # CONVERT MASK TO IDXS
                        NEW_MASK = np.empty(len(GIVEN_MASK), dtype=object)
                        for col_idx in range(len(GIVEN_MASK)):
                            NEW_MASK[col_idx] = np.nonzero(GIVEN_MASK[col_idx])[-1]
                        # END CONVERT MASK

                        if given_orientation == 'ROW':
                            if mask_format[0] == '2':
                                EXPECTED = (sd.delete_outer_key(GIVEN_OBJECT, NEW_MASK[0])[0],
                                            sd.multi_select_outer(GIVEN_OBJECT, NEW_MASK[0]))
                            elif mask_format[0] == '3':
                                EXPECTED = (sd.delete_outer_key(GIVEN_OBJECT, sorted(np.hstack((NEW_MASK[0], NEW_MASK[1])).tolist()))[0],
                                            sd.multi_select_outer(GIVEN_OBJECT, NEW_MASK[0]),
                                            sd.multi_select_outer(GIVEN_OBJECT, NEW_MASK[1]))
                        elif given_orientation == 'COLUMN':
                            if mask_format[0] == '2':
                                EXPECTED = (sd.delete_inner_key(GIVEN_OBJECT, NEW_MASK[0])[0],
                                            sd.multi_select_inner(GIVEN_OBJECT, NEW_MASK[0]))
                            elif mask_format[0] == '3':
                                EXPECTED = (sd.delete_inner_key(GIVEN_OBJECT, sorted(np.hstack((NEW_MASK[0], NEW_MASK[1])).tolist()))[0],
                                            sd.multi_select_inner(GIVEN_OBJECT, NEW_MASK[0]),
                                            sd.multi_select_inner(GIVEN_OBJECT, NEW_MASK[1]))

                    # SEE IF ANY NON-INTEGERS IN ACTUAL/EXPECTED OBJECTS #################################################
                    for obj_idx in range(len(ACTUAL)):
                        if object_format[:2]=='np':
                            # TEST DTYPE OF NP OBJECT CONTAINS "INT"
                            if 'INT' not in str(ACTUAL[obj_idx].dtype).upper():
                                test_except(f'ACTUAL NP ARRAY IS NOT DTYPE INT')
                            if 'INT' not in str(EXPECTED[obj_idx].dtype).upper():
                                test_except(f'EXPECTED NP ARRAY IS NOT DTYPE INT')

                        elif object_format[:2]=='sd':
                            # TEST SD PASSES IS_SPARSE_DICT TEST
                            if not sd.is_sparse_dict(ACTUAL[obj_idx]):
                                [print(f'{_}: {ACTUAL[obj_idx][_]}') for _ in ACTUAL[obj_idx]]
                                test_except(f'ACTUAL SPARSE DICT DOES NOT CONFORM TO SPARSE DICT FORMAT RULES')
                            if not sd.is_sparse_dict(EXPECTED[obj_idx]):
                                test_except(f'EXPECTED SPARSE DICT DOES NOT CONFORM TO SPARSE DICT FORMAT RULES')

                    # END SEE IF ANY NON-INTEGERS IN ACTUAL/EXPECTED OBJECTS ##################################################

                    if len(ACTUAL) != len(EXPECTED):
                        test_except(f'len(ACTUAL) ({len(ACTUAL)}) DOES NOT MATCH len(EXPECTED) ({len(EXPECTED)})')

                    for output_idx in range(len(EXPECTED)):
                        print(f'\033[91m')
                        if not np.array_equiv(ACTUAL[output_idx], EXPECTED[output_idx]):
                            print(f'\n *** EXPECTED & OUTPUT IDX {output_idx} ARE NOT EQUAL***')
                            print(f'GIVEN_OBJECT:')
                            [print(f'{_}:{GIVEN_OBJECT[_]}') for _ in GIVEN_OBJECT]
                            print()
                            print(f'EXPECTED:')
                            [print(_, f'\n') for _ in EXPECTED]
                            print()
                            print(f'ACTUAL:')
                            [print(_, f'\n') for _ in ACTUAL]
                            test_except('')
                        print(f'\033[0m')


    print(f'\033[92mALL TESTS PASSED\033[0m')
    for _ in range(3): wls.winlinsound(888,500); time.sleep(1)






