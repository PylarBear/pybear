import sys, inspect
from copy import deepcopy
import numpy as n, sparse_dict as sd
from data_validation import validate_user_input as vui, arg_kwarg_validater as akv
from ML_PACKAGE._data_validation import list_dict_validater as ldv
from debug import get_module_name as gmn
from general_data_ops import get_shape as gs


# RETURNS A MASK AND A CONTEXT UPDATE TO ALLOW FOR APPLICATION TO OLD OR NEW SUPOBJS


def split_str_cutoff_filter(DATA, data_orientation, HEADER, DATA_VALIDATED_DATATYPES, DATA_MODIFIED_DATATYPES,
                            DATA_MIN_CUTOFFS):
    # PASS VTYPES, MTYPES, MIN_CUTOFFS FOR DATA ONLY... IE, MODIFIED_DATATYPES[0]

    # BREAK ANY TIES TO A self THAT MAY HAVE BEEN PASSED
    DATA = deepcopy(DATA) if isinstance(DATA, dict) else DATA.copy()

    data_orientation = akv.arg_kwarg_validater(data_orientation.upper(),'data_orientation', ["ROW", "COLUMN"],
                                gmn.get_module_name(str(sys.modules[__name__])), inspect.stack()[0][3])

    HEADER = ldv.list_dict_validater(HEADER, 'HEADER')[1][0]

    data_given_format, DATA = ldv.list_dict_validater(DATA, 'DATA')
    is_list, is_dict = data_given_format=='ARRAY', data_given_format=='SPARSE_DICT'
    del data_given_format

    DATA_VALIDATED_DATATYPES = ldv.list_dict_validater(DATA_VALIDATED_DATATYPES, 'HEADER')[1][0]
    DATA_MODIFIED_DATATYPES = ldv.list_dict_validater(DATA_MODIFIED_DATATYPES, 'HEADER')[1][0]
    DATA_MIN_CUTOFFS = ldv.list_dict_validater(DATA_MIN_CUTOFFS, 'HEADER')[1][0]

    _rows, _columns = gs.get_shape('DATA', DATA, data_orientation)

    if data_orientation == 'ROW':
        if is_list: DATA = DATA.transpose()
        elif is_dict: DATA = sd.core_sparse_transpose(DATA)

    while True:
        MASK = n.empty(0, dtype=n.int32)    # MASK HOLDS COLUMN INDEXES TO BE DELETED (NOT KEPT)
        CONTEXT_UPDATE = []

        VTYPE_STR = n.fromiter(map(lambda x: x== 'STR', DATA_VALIDATED_DATATYPES), dtype=bool)
        MTYPE_INT = n.fromiter(map(lambda x: x == 'INT', DATA_MODIFIED_DATATYPES), dtype=bool)
        IS_SPLIT_STR = VTYPE_STR.astype(n.int8) * MTYPE_INT.astype(n.int8)
        del VTYPE_STR, MTYPE_INT

        # IF NO MATCHING PAIRS, max OF SPLIT_STR IS ZERO
        if max(IS_SPLIT_STR.astype(n.int8)) == 0:
            print(f'\n*** DATA HAS NO SPLIT STRING COLUMNS ***\n')
            break

        # TURN IS_SPLIT_STR INTO A VECTOR OF COLUMN INDEXES
        SPLIT_STR_COLS = n.nonzero(IS_SPLIT_STR)[-1].astype(n.int32).reshape((1,-1))[0]
        del IS_SPLIT_STR

        # IF THERE ARE SPLIT_STR COLUMNS AND NO MIN_CUTOFFS ARE SET, PROMPT USER FOR THRESHOLD
        if False not in map(lambda x: x==0, DATA_MIN_CUTOFFS[SPLIT_STR_COLS]):
            # DISPLAY FREQUENCIES #####################################################################################
            print(f'\nFREQUENCIES:')
            max_hdr_len = min(max(map(len, HEADER)) + 3,100)
            if is_list:
                for col_idx in SPLIT_STR_COLS:
                    UNIQUES_DICT = dict((zip(*n.unique(DATA[col_idx], return_counts=True))))
                    print(f'{HEADER[col_idx][:97]}'.ljust(max_hdr_len) + f'{_rows - UNIQUES_DICT[0]}')
                del UNIQUES_DICT
            elif is_dict:
                for col_idx in SPLIT_STR_COLS:  # JUST USE LEN OF SD COLUMN ADJUSTED FOR PLACEHOLDER TO GET NUM ENTRIES
                    ct = len(DATA[col_idx]) - int(DATA[col_idx][_rows - 1]==0)
                    print(f'{HEADER[col_idx][:97]}'.ljust(max_hdr_len) + f'{ct}')
                del ct
            # END DISPLAY FREQUENCIES #####################################################################################

            while True:
                cutoff = vui.validate_user_int(f'\nEnter min cutoff to apply to SPLIT_STR columns > ', min=0, max=_rows)
                if vui.validate_user_str(f'\nUser entered {cutoff}.... Accept? (y/n) > ', 'YN'): break
            DATA_MIN_CUTOFFS[SPLIT_STR_COLS] = cutoff

            del cutoff

        ####################################################################################################################
        # BUILD A MASK TO INDICATE WHICH COLUMNS TO DELETE FROM DATA AND SUPPORT OBJECTS #############################

        if is_list:
            for col_idx in SPLIT_STR_COLS:
                UNIQUES_DICT = dict((zip(*n.unique(DATA[col_idx], return_counts=True))))
                if (_rows - UNIQUES_DICT[0]) < DATA_MIN_CUTOFFS[col_idx]: MASK = n.hstack((MASK, col_idx))
            del UNIQUES_DICT
        elif is_dict:
            for col_idx in SPLIT_STR_COLS:  # JUST USE LEN OF SD COLUMN ADJUSTED FOR PLACEHOLDER TO GET NUM ENTRIES
                ct = len(DATA[col_idx]) - int(DATA[col_idx][_rows-1]==0)
                if ct < DATA_MIN_CUTOFFS[col_idx]: MASK = n.hstack((MASK, col_idx))

            del ct

        # END BUILD A MASK TO INDICATE WHICH COLUMNS TO DELETE FROM DATA AND SUPPORT OBJECTS #############################
        ####################################################################################################################

        if len(MASK) == 0:
            print(f'\n*** THERE ARE NO SPLIT_STR COLUMNS THAT FAIL THE CUTOFF CRITERIA ***\n')
            _ = vui.validate_user_str(f'\nTry again(t) or exit(e) > ', 'TE')
            if _ == 'T': DATA_MIN_CUTOFFS=None; continue
            elif _ == 'E': break

        # GET COLUMN NAMES FROM MASK & HEADER TO PROMPT USER
        COLUMN_NAMES = HEADER[MASK]

        print(f'\nGOING TO DELETE THE FOLLOWING COLUMNS FOR FAILING TO MEET SPLIT_STR CUTOFF CRITERIA:')
        [print(_) for _ in COLUMN_NAMES]

        _ = vui.validate_user_str(f'\nAccept(a), exit(e), try again(t) > ', 'AET')
        if _ == 'A': pass
        elif _ == 'E': MASK = n.empty(0, dtype=n.int32); DATA_MIN_CUTOFFS=None; break  # CONTEXT_UPDATE IS STILL EMPTY
        elif _ == 'T': continue

        # IF USER ACCEPTED, UPDATE CONTEXT, THEN RETURN MASK AND CONTEXT_UPDATE
        # PUT COLUMN NAMES AND DESCRIPTION INTO CONTEXT_UPDATE
        [CONTEXT_UPDATE.append(f'Deleted {name} for failing SPLIT_STR MIN CUTOFF ({cutoff})') for name, cutoff in zip(COLUMN_NAMES, DATA_MIN_CUTOFFS)]

        break

    del DATA

    return MASK, CONTEXT_UPDATE, DATA_MIN_CUTOFFS












if __name__ == '__main__':

    # TEST MODULE

    from MLObjects.TestObjectCreators.SXNL import CreateSXNL as csxnl

    data_return_format = 'ARRAY'
    data_return_orient = 'ROW'
    target_return_format = 'ARRAY'
    target_return_orient = 'COLUMN'
    refvecs_return_format = 'ARRAY',
    refvecs_return_orient = 'COLUMN',

    SXNLClass = csxnl.CreateSXNL(rows=1000,
                                 bypass_validation=False,
                                 data_return_format=data_return_format,
                                 data_return_orientation=data_return_orient,
                                 DATA_OBJECT=None,
                                 DATA_OBJECT_HEADER=None,
                                 DATA_FULL_SUPOBJ_OR_SINGLE_MDTYPES=['FLOAT','INT','SPLIT_STR', 'STR', 'INT', 'BIN'],
                                 data_override_sup_obj=False,
                                 data_given_orientation=None,
                                 data_columns=6,
                                 DATA_BUILD_FROM_MOD_DTYPES=None,
                                 DATA_NUMBER_OF_CATEGORIES=None,
                                 DATA_MIN_VALUES=0,
                                 DATA_MAX_VALUES=10,
                                 DATA_SPARSITIES=0,
                                 DATA_WORD_COUNT=10,
                                 DATA_POOL_SIZE=100,
                                 target_return_format=target_return_format,
                                 target_return_orientation=target_return_orient,
                                 TARGET_OBJECT=None,
                                 TARGET_OBJECT_HEADER=None,
                                 TARGET_FULL_SUPOBJ_OR_SINGLE_MDTYPES=None,
                                 target_type='FLOAT',
                                 target_override_sup_obj=False,
                                 target_given_orientation=None,
                                 target_sparsity=None,
                                 target_build_from_mod_dtype='FLOAT',
                                 target_min_value=-10,
                                 target_max_value=10,
                                 target_number_of_categories=None,
                                 refvecs_return_format=refvecs_return_format,
                                 refvecs_return_orientation=refvecs_return_orient,
                                 REFVECS_OBJECT=None,
                                 REFVECS_OBJECT_HEADER=[['ROW_ID']],
                                 REFVECS_FULL_SUPOBJ_OR_SINGLE_MDTYPES=None,
                                 REFVECS_BUILD_FROM_MOD_DTYPES='STR',
                                 refvecs_override_sup_obj=False,
                                 refvecs_given_orientation=None,
                                 refvecs_columns=3,
                                 REFVECS_NUMBER_OF_CATEGORIES=10,
                                 REFVECS_MIN_VALUES=None,
                                 REFVECS_MAX_VALUES=None,
                                 REFVECS_SPARSITIES=None,
                                 REFVECS_WORD_COUNT=None,
                                 REFVECS_POOL_SIZE=None
                                 )

    SRNL = SXNLClass.SXNL.copy()
    RAW_SUPOBJS = SXNLClass.SXNL_SUPPORT_OBJECTS.copy()

    SXNLClass.expand_data(expand_as_sparse_dict=False, auto_drop_rightmost_column=False)

    SWNL = SXNLClass.SXNL
    WORKING_SUPOBJS = SXNLClass.SXNL_SUPPORT_OBJECTS
    del SXNLClass


    MASK, CONTEXT_UPDATE, MIN_CUTOFFS = split_str_cutoff_filter(SWNL[0], data_return_orient, WORKING_SUPOBJS[0][0],
                                                  WORKING_SUPOBJS[0][1], WORKING_SUPOBJS[0][2], WORKING_SUPOBJS[0][4])

    print(f'\nMASK:')
    print(MASK)
    print()
    print(f'\nCONTEXT_UPDATE:')
    print(CONTEXT_UPDATE)
    print()
    print(f'\nMIN_CUTOFFS:')
    print(MIN_CUTOFFS)





