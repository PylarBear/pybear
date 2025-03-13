import numpy as np, sparse_dict as sd
from data_validation import validate_user_input as vui, arg_kwarg_validater as akv


def _exception(text_string):
    raise Exception(f'\n*** {text_string} ***\n')


def get_uniques(OBJECT):
    UNIQUES = np.empty(len(OBJECT), dtype=object)  # OBJECT IS ORIENTED AS COLUMN BY THIS POINT
    for idx in range(len(OBJECT)):
        # 11/28/22 W/O return_index, UNIQUES COMES BACK SORTED ALPHABETICALLY OR NUMERICALLY, USE return_index TO GET ORDER IN ORIGINAL DATA
        INDICES_OF_UNIQUES = np.unique(OBJECT[idx], return_index=True)[1]
        UNIQUES[idx] = OBJECT[idx][sorted(INDICES_OF_UNIQUES)]

    del INDICES_OF_UNIQUES
    return UNIQUES


def get_dummies(OBJECT_AS_LISTTYPE, OBJECT_HEADER=None, given_orientation='COLUMN', IDXS=None, UNIQUES=None,
                return_orientation='COLUMN', expand_as_sparse_dict=False, auto_drop_rightmost_column=False,
                append_ones=False, bypass_validation=False, bypass_sum_check=True, calling_module=None, calling_fxn=None):
    '''RETURNS EXPANDED_OBJECT, EXPANDED_OBJECT_HEADER, DROPPED_COLUMN_NAMES; DTYPES ARE NOT RETURNED BECAUSE THEY ARE ALWAYS 'BIN'.
    Processed as [] = columns. This module blindly expands given columns of any dtype into levels, including float and
    integer columns. To avoid expanding unwanted columns, the columns must be deselected out of "OBJECT" before passing
    to this function or through omission from "IDX" kwarg. Can bring in UNIQUES as arg for speed; np.uniques is expensive.'''

    # TO USE bypass, OBJECT WILL BE PROCESSED AS [] = COLUMN AND VECTORS IN UNIQUES MUST MATCH VECTORS IN OBJECT
    if not isinstance(bypass_validation, bool):
        _exception(f'bypass_validation MUST BE A BOOLEAN')

    if not bypass_validation:
        # VALIDATION #################################################################################################
        _ = calling_module if not calling_module is None else 'general_data_ops.get_dummies'
        __ = calling_fxn if not calling_fxn is None else 'get_dummies'
        akv.arg_kwarg_validater(given_orientation, 'given_orientation', ['COLUMN', 'ROW'], _, __)
        akv.arg_kwarg_validater(return_orientation, 'return_orientation', ['COLUMN', 'ROW'], _, __)
        akv.arg_kwarg_validater(expand_as_sparse_dict, 'expand_as_sparse_dict', [True, False], _, __)
        akv.arg_kwarg_validater(auto_drop_rightmost_column, 'auto_drop_rightmost_column', [True, False], _, __)
        akv.arg_kwarg_validater(append_ones, 'append_ones',  [True, False], _, __)
        akv.arg_kwarg_validater(bypass_validation, 'bypass_validation', [True, False], _, __)
        akv.arg_kwarg_validater(bypass_sum_check, 'bypass_sum_check', [True, False], _, __)
        del _, __, calling_module, calling_fxn

        if OBJECT_AS_LISTTYPE is None:
            _exception(f'"OBJECT_AS_LISTTYPE" MUST BE A LIST-TYPE THAT CAN CONVERT TO AN ndarray, CANNOT BE None')
        else:
            try:
                OBJECT_AS_LISTTYPE = np.array(OBJECT_AS_LISTTYPE)
                if len(OBJECT_AS_LISTTYPE.shape) == 1: OBJECT = OBJECT_AS_LISTTYPE.reshape((1,-1))
                else: OBJECT = OBJECT_AS_LISTTYPE
            except:
                _exception(f'"OBJECT_AS_LISTTYPE" MUST BE A LIST-TYPE THAT CAN CONVERT TO AN ndarray, NOT ""')

        # PUT OBJECT INTO [ [] = COLUMN ]
        if given_orientation == 'ROW':  OBJECT = OBJECT.transpose()

        EXPANDED_OBJECT_HEADER = np.empty((1,0), dtype='<U200')
        if not OBJECT_HEADER is None:
            has_header = True
            try: OBJECT_HEADER = np.array(OBJECT_HEADER).reshape((1,-1))
            except: _exception(f'"OBJECT_HEADER" MUST BE A LIST-TYPE THAT CAN CONVERT TO AN ndarray')

            if len(OBJECT_HEADER[0]) != len(OBJECT):
                _exception(f'Header len ({len(OBJECT_HEADER[0])}) DOES NOT MATCH NUMBER OF COLUMNS ({len(OBJECT)}) FOR '
                           f'given_orientation ({given_orientation})')
        else:
            has_header = False

        if not IDXS is None:  # MUST BE AFTER OBJECT IS ORIENTED TO [ [] = COLUMNS ]
            try: IDXS = np.array(IDXS, dtype=np.int32).reshape((1, -1))[0]
            except: _exception(f'"IDXS" MUST BE A LIST-TYPE OF INTEGERS THAT CAN CONVERT TO AN ndarray')
            if len(IDXS) == 0: _exception(f'"IDXS" MUST HAVE INTEGERS, CANNOT BE EMPTY')
            if np.max(IDXS) >= len(OBJECT): _exception(f'MAX IDX IN IDXS ({np.max(IDXS)}) IS OUT OF RANGE OF OBJECT ({len(OBJECT)-1})')
            if np.min(IDXS) < 0: _exception(f'MIN IDX IN IDXS ({np.min(IDXS)}) IS OUT OF RANGE OF OBJECT.')
        elif IDXS is None: IDXS = range(len(OBJECT))

        if not UNIQUES is None:
            # UNIQUES PROBABLY WILL BE RAGGED
            try:
                if not isinstance(UNIQUES[0], (np.ndarray, list, tuple)): UNIQUES = np.array(UNIQUES).reshape((1,-1))
            except: _exception(f'"UNIQUES" MUST BE A LIST-TYPE THAT CAN CONVERT TO AN ndarray')

            if len(UNIQUES) != len(OBJECT):
                _exception(f'NUMBER OF COLUMNS IN UNIQUES ({len(UNIQUES)}) DOES NOT MATCH COLUMNS IN OBJECT ({len(OBJECT)})')

            for idx in range(len(UNIQUES)):
                for unique in UNIQUES[idx]:
                    if unique not in OBJECT[idx].astype('<U10000'):
                        _exception(f'"{unique}" IN UNIQUES COLUMN "{OBJECT_HEADER[0][idx]}" IS NOT IN RESPECTIVE OBJECT COLUMN')

        elif UNIQUES is None:
            UNIQUES = get_uniques(OBJECT)
        # END VALIDATION #################################################################################################

    elif bypass_validation:
        EXPANDED_OBJECT_HEADER = np.empty((1, 0), dtype='<U500')
        OBJECT = np.array(OBJECT_AS_LISTTYPE)
        if given_orientation == 'ROW':  OBJECT = OBJECT.transpose()
        if not OBJECT_HEADER is None: OBJECT_HEADER = np.array(OBJECT_HEADER).reshape((1,-1)); has_header = True
        else: has_header = False

        # MUST BE AFTER OBJECT IS ORIENTED TO [ [] = COLUMNS ]
        if not IDXS is None: IDXS = np.array(IDXS, dtype=np.int32).reshape((1, -1))[0]
        elif IDXS is None: IDXS = np.fromiter(range(len(OBJECT)), dtype=np.int32)

        if UNIQUES is None: UNIQUES = get_uniques(OBJECT)
        elif len(UNIQUES) != len(OBJECT):
            _exception(f'NUMBER OF COLUMNS IN UNIQUES ({len(UNIQUES)}) DOES NOT MATCH COLUMNS IN OBJECT ({len(OBJECT)}) ')

    del OBJECT_AS_LISTTYPE, given_orientation, bypass_validation

    _len = len(OBJECT[0])
    dum_indicator = ' - '

    if not expand_as_sparse_dict: EXPANDED_OBJECT = np.empty((0, _len), dtype=np.int8)
    elif expand_as_sparse_dict: EXPANDED_OBJECT = {}

    DROPPED_COLUMN_NAMES = []

    # END "init" ########################################################################################################
    #####################################################################################################################
    #####################################################################################################################

    ctr = 0
    for col_idx in range(len(OBJECT)):
        if col_idx not in IDXS:
            '''    ORIGINALLY INTENDED TO HAVE COLUMNS NOT IN "IDXS" TO JUST CARRY OVER TO RETURNED OBJECT, BUT IF A STR
                COLUMN, WILL BLOW UP A SPARSEDICT OR CONVERT AN ENTIRE NDARRAY DTYPE TO STR. SO NOW SIMPLY LEAVING OUT 
                COLUMNS NOT IN IDXS 
            # IF NOT IN IDXS, SIMPLY CARRY OVER TO EXPANDED OBJECT
            # APPEND COLUMN TO FULL EXPANDED HOLDER #####################################################################
            if not expand_as_sparse_dict:
                EXPANDED_OBJECT = np.vstack((EXPANDED_OBJECT, OBJECT[col_idx]))
            elif expand_as_sparse_dict:
                EXPANDED_OBJECT[int(len(EXPANDED_OBJECT))] = sd.zip_list_as_py_float(OBJECT[col_idx].reshape((1,-1)))[0]
            # END APPEND COLUMN TO FULL EXPANDED HOLDER #####################################################################

            # EXPAND HEADER ################################################################################################
            EXPANDED_OBJECT_HEADER = np.hstack((
                                                EXPANDED_OBJECT_HEADER,
                                                [[f'{OBJECT_HEADER[0][col_idx] if has_header else "COLUMN" + str(col_idx + 1)}']]
                                                ))
            ################################################################################################################
            '''

            continue   # 11/29/22 NEW METHOD - IF COLUMN NOT IN IDXS, LEAVE OUT OF RETURNED OBJECT

        elif col_idx in IDXS:
            if not expand_as_sparse_dict: LEVELS = np.zeros((len(UNIQUES[col_idx]), _len), dtype=np.int8)
            elif expand_as_sparse_dict: LEVELS = {int(_): {} for _ in range(len(UNIQUES[col_idx]))}

            # SPLIT OBJECT[column] INTO LEVELS #################################################################################
            HITS = None  # KEEP THIS TO AVOID BLOWUP ON del
            for unique_idx in range(len(UNIQUES[col_idx])):
                ctr += 1

                if ctr % 100 == 0:  # PRINT PROGRESS
                    print(f'Working on column {ctr} of {UNIQUES.size}...')

                HITS = np.where(OBJECT[col_idx].astype(str) == str(UNIQUES[col_idx][unique_idx]), 1, 0).astype(np.int8)

                if not expand_as_sparse_dict:
                    LEVELS[unique_idx] = HITS

                elif expand_as_sparse_dict:
                    NON_ZEROS = np.nonzero(HITS)[-1]
                    LEVELS[int(unique_idx)] = dict((zip(NON_ZEROS.tolist(), (int(1) for _ in NON_ZEROS))))
                    if _len-1 not in LEVELS[unique_idx]: LEVELS[int(unique_idx)][int(_len-1)] = int(0)
                    del NON_ZEROS
            # END SPLIT OBJECT[column] INTO LEVELS #############################################################################

            del HITS

            # CHECK ROW SUMS ###############################################################################################
            if not bypass_sum_check:
                # CHECK SUM TO 1 FOR EACH GROUP OF LEVELS, IF CALLED FOR
                if not expand_as_sparse_dict:
                    print(f'Running sums-to-one checks for ndarrays...')
                    ROW_SUMS = np.sum(LEVELS, axis=0)
                elif expand_as_sparse_dict:
                    print(f'Running sums-to-one checks for sparse dicts. Patience...')
                    ROW_SUMS = np.zeros((1, _len), dtype=np.int8)[0]
                    for row_idx in range(_len):
                        ROW_SUMS[row_idx] = sd.sum_over_inner_key(LEVELS, row_idx)

                if np.min(ROW_SUMS) != 1 and np.max(ROW_SUMS) != 1:
                    print(f'\n*** LEVEL EXPANSION HAS AT LEAST ONE EXAMPLE THAT IS NOT SUMMING TO 1. ***')
                    LEV_ROW_IDX = np.nonzero(np.int8(ROW_SUMS != 1))[-1].astype(np.int32)
                    print(f'\nColumn is ' + [f'{OBJECT_HEADER[0][col_idx]}. ' if has_header else f'index {col_idx}.'][0])
                    print(f'\nRow index(es) in current data is {", ".join(LEV_ROW_IDX.astype(str))}.')
                    print(f'\nValues is/are: \n{", ".join(OBJECT[col_idx][LEV_ROW_IDX])}')
                    print(f'\nUnique categories are: \n{", ".join(UNIQUES[col_idx])}')
                    handle_levels = vui.validate_user_str \
                        (f'Terminate(t) or continue anyway(c) or continue and bypass sum check(s)? > ', 'TCS')
                    if handle_levels == 'T':
                        _exception(f'Categorical column expansion terminated by user for row sum != 1.')
                    elif handle_levels == 'C':
                        pass
                    elif handle_levels == 'S':
                        bypass_sum_check = True
                        continue

                del ROW_SUMS
                try : del LEV_ROW_IDX, handle_levels
                except: pass
            # END CHECK ROW SUMS ###############################################################################################

            # APPEND EXPANDED LEVELS TO FULL EXPANDED HOLDER #####################################################################

            if not expand_as_sparse_dict: EXPANDED_OBJECT = np.vstack((EXPANDED_OBJECT,
                                                        LEVELS if not auto_drop_rightmost_column else LEVELS[:-1]))
            elif expand_as_sparse_dict:
                for level_col_idx in range(len(LEVELS) if not auto_drop_rightmost_column else len(LEVELS)-1):
                    EXPANDED_OBJECT[int(len(EXPANDED_OBJECT))] = LEVELS[level_col_idx]

                # 11/30/22 BEAR merge_outer ISNT WORKING RIGHT. THINKING IT CANT MERGE WHEN AN EMPTY DICT IS INVOLVED.
                # JUST SOMETHING ELSE TO FIX
                # sd.merge_outer(EXPANDED_OBJECT, LEVELS if not auto_drop_rightmost_column else LEVELS[:-1])

            # END APPEND EXPANDED LEVELS TO FULL EXPANDED HOLDER #####################################################################

            # EXPAND HEADER #########################################################################################################


            EXPANDED_OBJECT_HEADER = np.hstack((
                EXPANDED_OBJECT_HEADER,
                np.fromiter(
                    (f'{OBJECT_HEADER[0][col_idx] if has_header else "COLUMN"+str(col_idx+1)}{dum_indicator}{_}' for _ in
                     [UNIQUES[col_idx] if not auto_drop_rightmost_column else UNIQUES[col_idx][:-1]][0]),
                    dtype='<U500'
                ).reshape((1,-1))
            ))

            if auto_drop_rightmost_column: DROPPED_COLUMN_NAMES.append(
                f'{[OBJECT_HEADER[0][col_idx] if has_header else "COLUMN" + str(col_idx + 1)][0]}{dum_indicator}{UNIQUES[col_idx][-1]}'
                )
            #####################################################################################################################

    del IDXS, OBJECT, OBJECT_HEADER, LEVELS, UNIQUES, auto_drop_rightmost_column, bypass_sum_check, has_header

    if append_ones:
        if not expand_as_sparse_dict: EXPANDED_OBJECT = np.insert(EXPANDED_OBJECT, len(EXPANDED_OBJECT), 1, axis=0)
        elif expand_as_sparse_dict: EXPANDED_OBJECT = sd.append_outer(EXPANDED_OBJECT,
                                                                  np.fromiter((int(1) for _ in range(_len)), dtype=np.int8))

        EXPANDED_OBJECT_HEADER = np.hstack((EXPANDED_OBJECT_HEADER, [['ONES']]))

    if return_orientation == 'ROW':
        if not expand_as_sparse_dict: EXPANDED_OBJECT = EXPANDED_OBJECT.transpose()
        elif expand_as_sparse_dict: EXPANDED_OBJECT = sd.sparse_transpose(EXPANDED_OBJECT)

    del return_orientation, expand_as_sparse_dict, append_ones

    return EXPANDED_OBJECT, EXPANDED_OBJECT_HEADER, DROPPED_COLUMN_NAMES





if __name__ == '__main__':

    # TEST MODULE
    dum_indicator = ' - '
    OBJECT_AS_LISTTYPE = [['A', 'B', 'A'], ['C', 'B', 'E'], ['F','F','A']]

    HEADER = np.array(['X', 'Y', 'Z'], dtype='<U1').reshape((1,-1))

    UNIQUES_AS_COLUMN = [['A','B'],['C','B','E'],['F','A']]
    UNIQUES_AS_ROW = [['A','C','F'], ['B', 'F'], ['A', 'E']]

    # W/O DROP RIGHTMOST
    ANSWER_OBJECT_COLUMN_ARRAY_RETURN_COLUMN = np.array([[1,0,1],[0,1,0],[1,0,0],[0,1,0],[0,0,1],[1,1,0],[0,0,1]])
    ANSWER_OBJECT_COLUMN_ARRAY_RETURN_ROW = np.array([[1, 0, 1, 0, 0, 1, 0], [0, 1, 0, 1, 0, 1, 0], [1, 0, 0, 0, 1, 0, 1]])
    ANSWER_OBJECT_ROW_ARRAY_RETURN_COLUMN = np.array([[1,0,0],[0,1,0],[0,0,1],[1,1,0],[0,0,1],[1,0,1],[0,1,0]])
    ANSWER_OBJECT_ROW_ARRAY_RETURN_ROW = np.array([[1,0,0,1,0,1,0],[0,1,0,1,0,0,1],[0,0,1,0,1,1,0]])
    ANSWER_HEADER_NONE_COLUMN = np.array([f'COLUMN1{dum_indicator}A', f'COLUMN1{dum_indicator}B', f'COLUMN2{dum_indicator}C', f'COLUMN2{dum_indicator}B', f'COLUMN2{dum_indicator}E', f'COLUMN3{dum_indicator}F', f'COLUMN3{dum_indicator}A'], dtype='<U100').reshape((1,-1))
    ANSWER_HEADER_NONE_ROW = np.array([f'COLUMN1{dum_indicator}A',f'COLUMN1{dum_indicator}C', f'COLUMN1{dum_indicator}F', f'COLUMN2{dum_indicator}B',f'COLUMN2{dum_indicator}F',f'COLUMN3{dum_indicator}A',f'COLUMN3{dum_indicator}E'], dtype='<U100').reshape((1,-1))
    ANSWER_HEADER_COLUMN = np.array([f'X{dum_indicator}A', f'X{dum_indicator}B', f'Y{dum_indicator}C', f'Y{dum_indicator}B', f'Y{dum_indicator}E', f'Z{dum_indicator}F', f'Z{dum_indicator}A'], dtype='<U100').reshape((1,-1))
    ANSWER_HEADER_ROW = np.array([f'X{dum_indicator}A',f'X{dum_indicator}C', f'X{dum_indicator}F', f'Y{dum_indicator}B',f'Y{dum_indicator}F',f'Z{dum_indicator}A',f'Z{dum_indicator}E'], dtype='<U100').reshape((1,-1))

    # W DROP RIGHTMOST
    ANSWER_OBJECT_COLUMN_ARRAY_RETURN_COLUMN_DROP_RIGHT = np.array([[1,0,1],[1,0,0],[0,1,0],[1,1,0]])
    ANSWER_OBJECT_COLUMN_ARRAY_RETURN_ROW_DROP_RIGHT = np.array([[1, 1, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0]])
    ANSWER_OBJECT_ROW_ARRAY_RETURN_COLUMN_DROP_RIGHT = np.array([[1,0,0],[0,1,0],[1,1,0],[1,0,1]])
    ANSWER_OBJECT_ROW_ARRAY_RETURN_ROW_DROP_RIGHT = np.array([[1,0,1,1],[0,1,1,0],[0,0,0,1]])
    ANSWER_HEADER_NONE_COLUMN_DROP_RIGHT = np.array([f'COLUMN1{dum_indicator}A', f'COLUMN2{dum_indicator}C', f'COLUMN2{dum_indicator}B', f'COLUMN3{dum_indicator}F'], dtype='<U100').reshape((1,-1))
    ANSWER_HEADER_NONE_ROW_DROP_RIGHT = np.array([f'COLUMN1{dum_indicator}A',f'COLUMN1{dum_indicator}C', f'COLUMN2{dum_indicator}B',f'COLUMN3{dum_indicator}A'], dtype='<U100').reshape((1,-1))
    ANSWER_HEADER_COLUMN_DROP_RIGHT = np.array([f'X{dum_indicator}A', f'Y{dum_indicator}C', f'Y{dum_indicator}B', f'Z{dum_indicator}F'], dtype='<U100').reshape((1,-1))
    ANSWER_HEADER_ROW_DROP_RIGHT = np.array([f'X{dum_indicator}A',f'X{dum_indicator}C', f'Y{dum_indicator}B',f'Z{dum_indicator}A'], dtype='<U100').reshape((1,-1))






    MASTER_OBJECT_HEADER = [None, 'built on the fly']
    MASTER_UNIQUES = [None, 'built on the fly']
    GIVEN_ORIENTATION = ['COLUMN', 'ROW']
    RETURN_ORIENTATION = ['COLUMN', 'ROW']
    EXPAND_AS_SPARSE_DICT = [False, True]
    AUTO_DROP_RIGHTMOST_COLUMN = [True, False]
    APPEND_ONES = [True, False]
    BYPASS_VALIDATION = [False, True]
    BYPASS_SUM_CHECK = [True, False]

    total_trials = np.product(list(map(len, (MASTER_OBJECT_HEADER, MASTER_UNIQUES, GIVEN_ORIENTATION,
                        RETURN_ORIENTATION, EXPAND_AS_SPARSE_DICT, AUTO_DROP_RIGHTMOST_COLUMN, APPEND_ONES,
                        BYPASS_VALIDATION, BYPASS_SUM_CHECK))))

    print(f'Total trials = {total_trials}')

    ctr = 0
    for OBJECT_HEADER in MASTER_OBJECT_HEADER:
        for UNIQUES in MASTER_UNIQUES:
            for given_orientation in GIVEN_ORIENTATION:
                for return_orientation in RETURN_ORIENTATION:
                    for expand_as_sparse_dict in EXPAND_AS_SPARSE_DICT:
                        for auto_drop_rightmost_column in AUTO_DROP_RIGHTMOST_COLUMN:
                            for append_ones in APPEND_ONES:
                                for bypass_validation in BYPASS_VALIDATION:
                                    for bypass_sum_check in BYPASS_SUM_CHECK:
                                        ctr += 1

                                        # SET CONDITIONAL INPUTS ##################################################################
                                        if not OBJECT_HEADER is None:
                                            OBJECT_HEADER = HEADER

                                        if not UNIQUES is None:
                                            if given_orientation == 'COLUMN':
                                                UNIQUES = UNIQUES_AS_COLUMN
                                            elif given_orientation == 'ROW':
                                                UNIQUES = UNIQUES_AS_ROW
                                        # END SET CONDITIONAL INPUTS ##################################################################

                                        print(f'*****************************************************************************')
                                        print(f'\nRunning trial {ctr} of {total_trials} trials...')
                                        print(f'\nTesting')
                                        print(f'OBJECT_AS_LISTTYPE = 🤬')
                                        print(f'OBJECT_HEADER = {OBJECT_HEADER}')
                                        print(f'UNIQUES = {UNIQUES}')
                                        print(f'given_orientation = {given_orientation}')
                                        print(f'return_orientation = {return_orientation}')
                                        print(f'expand_as_sparse_dict = {expand_as_sparse_dict}')
                                        print(f'auto_drop_rightmost_column = {auto_drop_rightmost_column}')
                                        print(f'append_ones = {append_ones}')
                                        print(f'bypass_validation = {bypass_validation}')
                                        print(f'bypass_sum_check = {bypass_sum_check}')

                                        # PROVED THAT
                                        # catches OBJECT is None
                                        # catches bad arg in given_orientation
                                        # catches bad arg in return_orientation
                                        # catches bad arg in expand_as_sparse_dict
                                        # catches bad arg in auto_drop_rightmost_column
                                        # catches bad arg in append_ones
                                        # catches bad arg in bypass_validation
                                        # catches bad arg in bypass_sum_check

                                        # catches mismatch of len(HEADER) and len(OBJECT) when given_orient == 'COLUMN'
                                        # catches mismatch of len(HEADER) and len(OBJECT) when given_orient == 'ROW'
                                        # catches mismatch of len(UNIQUES) and len(OBJECT) when given_orient == 'COLUMN'
                                        # catches mismatch of len(UNIQUES) and len(OBJECT) when given_orient == 'ROW'

                                        # catches mismatch of an item in UNIQUES[col] not in OBJECT[col]

                                        ACT_EXPANDED_OBJECT, ACT_EXPANDED_HEADER, ACT_DROPPED_COLUMN_NAMES = \
                                            get_dummies(OBJECT_AS_LISTTYPE,
                                                        OBJECT_HEADER=OBJECT_HEADER,
                                                        given_orientation=given_orientation,
                                                        IDXS=None,
                                                        UNIQUES=UNIQUES,
                                                        return_orientation=return_orientation,
                                                        expand_as_sparse_dict=expand_as_sparse_dict,
                                                        auto_drop_rightmost_column=auto_drop_rightmost_column,
                                                        append_ones=append_ones,
                                                        bypass_validation=bypass_validation,
                                                        bypass_sum_check=bypass_sum_check
                                                        )

                                        # GET EXPECTED OBJECT AND HEADER ######################################################
                                        if not OBJECT_HEADER is None:
                                            if given_orientation == 'COLUMN':
                                                EXP_HEADER = ANSWER_HEADER_COLUMN
                                                if auto_drop_rightmost_column:
                                                    EXP_HEADER = ANSWER_HEADER_COLUMN_DROP_RIGHT
                                            elif given_orientation == 'ROW':
                                                EXP_HEADER = ANSWER_HEADER_ROW
                                                if auto_drop_rightmost_column:
                                                    EXP_HEADER = ANSWER_HEADER_ROW_DROP_RIGHT
                                        elif OBJECT_HEADER is None:
                                            if given_orientation == 'COLUMN':
                                                EXP_HEADER = ANSWER_HEADER_NONE_COLUMN
                                                if auto_drop_rightmost_column:
                                                    EXP_HEADER = ANSWER_HEADER_NONE_COLUMN_DROP_RIGHT
                                            elif given_orientation == 'ROW':
                                                EXP_HEADER = ANSWER_HEADER_NONE_ROW
                                                if auto_drop_rightmost_column:
                                                    EXP_HEADER = ANSWER_HEADER_NONE_ROW_DROP_RIGHT


                                        if given_orientation == 'COLUMN':
                                            if return_orientation == 'COLUMN':
                                                EXP_OBJECT = ANSWER_OBJECT_COLUMN_ARRAY_RETURN_COLUMN
                                                if auto_drop_rightmost_column:
                                                    EXP_OBJECT = ANSWER_OBJECT_COLUMN_ARRAY_RETURN_COLUMN_DROP_RIGHT
                                            elif return_orientation == 'ROW':
                                                EXP_OBJECT = ANSWER_OBJECT_COLUMN_ARRAY_RETURN_ROW
                                                if auto_drop_rightmost_column:
                                                    EXP_OBJECT = ANSWER_OBJECT_COLUMN_ARRAY_RETURN_ROW_DROP_RIGHT
                                        elif given_orientation == 'ROW':
                                            if return_orientation == 'COLUMN':
                                                EXP_OBJECT = ANSWER_OBJECT_ROW_ARRAY_RETURN_COLUMN
                                                if auto_drop_rightmost_column:
                                                    EXP_OBJECT = ANSWER_OBJECT_ROW_ARRAY_RETURN_COLUMN_DROP_RIGHT
                                            elif return_orientation == 'ROW':
                                                EXP_OBJECT = ANSWER_OBJECT_ROW_ARRAY_RETURN_ROW
                                                if auto_drop_rightmost_column:
                                                    EXP_OBJECT = ANSWER_OBJECT_ROW_ARRAY_RETURN_ROW_DROP_RIGHT

                                        if append_ones:
                                            EXP_HEADER = np.hstack((EXP_HEADER, [['ONES']]))

                                            EXP_OBJECT = np.insert(EXP_OBJECT,
                                                                   len(EXP_OBJECT) if return_orientation == 'COLUMN' else len(EXP_OBJECT[0]),
                                                                   1,
                                                                   axis=0 if return_orientation == 'COLUMN' else 1)

                                        if expand_as_sparse_dict:
                                            EXP_OBJECT = sd.zip_list_as_py_int(EXP_OBJECT)


                                        # END GET EXPECTED OBJECT AND HEADER ######################################################

                                        # PRINT EXPECTED AND ACTUALS TO SCREEN #####################################################
                                        print(f'\nEXP OBJECT:')
                                        print(EXP_OBJECT)
                                        print(f'\nEXP HEADER:')
                                        print(EXP_HEADER)
                                        print()

                                        print(f'\nACTUAL OBJECT:')
                                        print(ACT_EXPANDED_OBJECT)
                                        print(f'\nACTUAL HEADER:')
                                        print(ACT_EXPANDED_HEADER)
                                        print(f'\nACTUAL DROPPED COLUMN NAMES:')
                                        print(ACT_DROPPED_COLUMN_NAMES)
                                        # END PRINT EXPECTED AND ACTUALS TO SCREEN #####################################################

                                        # OUTPUT VALIDATION ####################################################################
                                        if expand_as_sparse_dict:
                                            if not sd.core_sparse_equiv(ACT_EXPANDED_OBJECT, EXP_OBJECT):
                                                raise Exception(f'ACT OBJECT AS SD != EXP OBJECT')
                                        elif not expand_as_sparse_dict:
                                            if not np.array_equiv(ACT_EXPANDED_OBJECT, EXP_OBJECT):
                                                raise Exception(f'ÁCT OBJECT AS NP != EXP OBJECT')

                                        if not np.array_equiv(ACT_EXPANDED_HEADER, EXP_HEADER):
                                            raise Exception(f'ÁCT HEADER AS NP != EXP HEADER')

                                        if not auto_drop_rightmost_column:
                                            if not np.array_equiv(ACT_DROPPED_COLUMN_NAMES, []):
                                                raise Exception(f'ÁCT DROPPED_COLUMN_NAMES SHOULD BE EMPTY AND IS NOT')
                                        elif auto_drop_rightmost_column:
                                            # COP OUT VALIDATION, ONLY CHECKING len
                                            if len(ACT_DROPPED_COLUMN_NAMES) != [len(OBJECT_AS_LISTTYPE) if given_orientation == 'COLUMN' else len(OBJECT_AS_LISTTYPE[0])][0]:
                                                raise Exception(f'ÁCT DROPPED_COLUMN_NAMES SHOULD BE len '
                                                                f'{len(OBJECT_AS_LISTTYPE) if given_orientation == "COLUMN" else len(OBJECT_AS_LISTTYPE[0])} '
                                                                f'BUT IS len {len(ACT_DROPPED_COLUMN_NAMES)}')
                                        # END OUTPUT VALIDATION ####################################################################

    print(f'\n\033[92m*** VALIDATION COMPLETED SUCCESSFULLY ***\033[0m\x1B[0m\n')












































