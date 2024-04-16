import numpy as n, pandas as p
from copy import deepcopy
from data_validation import validate_user_input as vui
from ML_PACKAGE.GENERIC_PRINT import show_time as st
import linear_algebra.matrix_transpose as mt

'''KEEP FOR REFERENCE
THIS IS NOW BEING HANDLED
IN MLRunTemplate'''




def svd_generic_test_run(standard_config, tc_method, TEST_OBJECT, TEST_OBJECT_HEADER, TARGET_OBJECT, TARGET_OBJECT_HEADER,
                        MODIFIED_DATATYPES, WINNING_COLUMNS, COEFFS):

    st.show_start_time('Calculate TEST CASES')
    print(f'Calculating TEST CASES for {tc_method}...')

    # PROCESS TEST_OBJECT & HEADER DOWN TO WINNING COLUMNS #############################################################
    TEST_OBJECT = n.array([TEST_OBJECT[_] for _ in WINNING_COLUMNS], dtype=object)
    TEST_OBJECT_HEADER[0] = [TEST_OBJECT_HEADER[0][_] for _ in WINNING_COLUMNS]
    MODIFIED_DATATYPES[0] = n.array([MODIFIED_DATATYPES[0][_] for _ in WINNING_COLUMNS], dtype=object)

    # CALCULATE SCORE VECTOR ###########################################################################################
    SCORE_VECTOR = n.matmul(TEST_OBJECT.astype(float), COEFFS.astype(float), dtype=float)

    #SORTING############################################################################################################
    # CREATE AN UN-NUMPY ARGSORT KEY FROM THE SCORES IN SCORE_VECTOR, KEY WILL BE USED ON TEST MATRIX AND SCORE_VECTOR
    # (reverse() WONT WORK ON A NUMPY)

    SCORE_VECTOR_SORT_KEY = [_ for _ in n.argsort(SCORE_VECTOR)[0]]  # argsort IS ASCENDING AT CONSTRUCTION
    if vui.validate_user_str(f'Sort ascending(a) or descending(d) > ', 'AD') == 'D':
        SCORE_VECTOR_SORT_KEY = list(reversed(SCORE_VECTOR_SORT_KEY))

    # SORT SCORE_VECTOR BY SORT KEY
    SORTED_SCORE_VECTOR = [SCORE_VECTOR[rank] for rank in SCORE_VECTOR_SORT_KEY]

    ####################################################################################################################
    #CONVERT NUMPY TEST MATRIX BACK TO REGULAR LISTS, AND CONVERT 1s TO CATEGORIES######################################
    #SORT UNNUMPY_TEST_MATRIX BY SORT KEY

    SORTED_UNNUMPY_TEST_MATRIX = [[] for _ in range(len(TEST_OBJECT))]
    for column_idx in range(len(TEST_OBJECT)):
        if MODIFIED_DATATYPES[column_idx] == 'FLOAT':
            for row_idx in range(len(TEST_OBJECT[column_idx])):
                SORTED_UNNUMPY_TEST_MATRIX[column_idx].append(TEST_OBJECT[column_idx][row_idx])

        elif MODIFIED_DATATYPES[column_idx] != 'FLOAT':
            for row_idx in range(len(TEST_OBJECT[column_idx])):
                locate_row = int(SCORE_VECTOR_SORT_KEY[row_idx])
                element = TEST_OBJECT[column_idx][locate_row]

                if element == 1: SORTED_UNNUMPY_TEST_MATRIX[column_idx].append(TEST_OBJECT_HEADER[0][column_idx])
                elif element == 0: SORTED_UNNUMPY_TEST_MATRIX[column_idx].append('')
    ####################################################################################################################

    # RECONSTRUCTING ORIGINAL DATA FROM THE (EXPANDED) DATA MATRIX

    # GENERATE A VECTOR IN TEST_OBJECT_HEADER_DUM THAT, FOR EACH RESPECTIVE FEATURE IN TEST_OBJECT_HEADER_DUM, ##################
    # HOLDS THAT FEATURES'S INDEX POSITION IN UNIQUE_ATTR LIST, ANOTHER VECTOR TO BE USED LATER IN BUILDING ################
    # CONDENSED_SORTED_UNNUMPY_TEST_MATRIX #############################################################################
    TEST_OBJECT_HEADER_DUM = deepcopy(TEST_OBJECT_HEADER)
    TEST_OBJECT_HEADER_DUM.append(['' for _ in TEST_OBJECT_HEADER_DUM[0]])

    index = 0
    UNIQUE_ATTR = []
    for attr_idx in range(len(TEST_OBJECT_HEADER_DUM[0])):

        attr = TEST_OBJECT_HEADER_DUM[0][attr_idx]

        if MODIFIED_DATATYPES[attr_idx] == 'FLOAT':    # IF IS FLOAT, JUST CARRY ALONG
            UNIQUE_ATTR.append(attr)
            index += 1
        else:
            # 2-15-22 BEAR THIS IS A COPOUT, JUST SPLITTING ON " - " FOR NOW TO GET FEATURES AND CATEGORIES.  IN THE FUTURE
            # HAVE TO FIGURE OUT A WAY TO DO THIS ROBUSTLY FOR LAG, INTERACTIONS, ETC.
            feature = attr[:attr.index(' - ')]
            if feature not in UNIQUE_ATTR:
                UNIQUE_ATTR.append(attr)
                index += 1

        TEST_OBJECT_HEADER_DUM[-1].append(index)


    ###########################################################################################################

    # CONSOLIDATE THE PREVIOUSLY EXPANDED RESULTS INTO SORTED_UNNUMPY_TEST_MATRIX (TAKE OUT BLANKS)#####################
    CSUTM = [] #CONDENSED_SORTED_UNNUMPY_TEST_MATRIX

    #BUILD A TEMPLATE THAT WILL BE USED TO DISTRIBUTE CATs CORRECTLY IN RELATION TO HEADER
    ROW_BLOCK = ['' for attr in UNIQUE_ATTR]

    for row_idx in range(len(SORTED_UNNUMPY_TEST_MATRIX[0])):
        NEW_ROW_BLOCK = ROW_BLOCK.copy()
        for column_idx in range(len(SORTED_UNNUMPY_TEST_MATRIX)):
            looked_up_index = TEST_OBJECT_HEADER_DUM[1][column_idx]
            if MODIFIED_DATATYPES[column_idx] == 'FLOAT':
                NEW_ROW_BLOCK[looked_up_index] = SORTED_UNNUMPY_TEST_MATRIX[column_idx][row_idx]
            else:
                if SORTED_UNNUMPY_TEST_MATRIX[column_idx][row_idx] != '':
                    NEW_ROW_BLOCK[looked_up_index] = \
                        SORTED_UNNUMPY_TEST_MATRIX[SORTED_UNNUMPY_TEST_MATRIX[column_idx][row_idx].index(' - ') + 3:]
        CSUTM.append(NEW_ROW_BLOCK)
    #############################################################################################################

    #CHECK IF SORTED SCORE VECTOR AND CONSOLIDATED, SORTED, UN-NUMPY-ED TEST MATRIX ARE SAME LENGTH
    if len(SORTED_SCORE_VECTOR) != len(CSUTM):
        raise AssertionError(f'len(SCORE_VECTOR) IS NOT THE SAME AS len(TEST_MATRIX) in generic_nn_test_run().')

    #ADD SCORE COLUMN TO CSUTM
    CSUTM = mt.matrix_transpose(CSUTM)
    CSUTM.append(deepcopy(SORTED_SCORE_VECTOR))
    CSUTM = mt.matrix_transpose(CSUTM)

    #CREATE ATTR HEADER FOR CONSOLIDATED_SORTED_UNNUMPY_TEST_MATRIX W/ SCORE COLUMN
    UNIQUE_ATTR.append('SCORE')

    CSUTM_DF = p.DataFrame(data=CSUTM, columns=UNIQUE_ATTR)

    # BUILDING OF RESULTS OBJECTS AND DISPLAY OBJECTS COMPLETE##########################################################

    print('\nTEST MATRIX results complete.')
    print(f'TEST MATRIX has {len(CSUTM)} test cases.')
    st.show_end_time('Calculate TEST CASES\n')

    print(f'CSUTM_DF[0,1,2][:20] AND ')
    print(CSUTM_DF[[[_] for _ in CSUTM_DF[:3]]].head(20))
    print(CSUTM_DF[[[_] for _ in CSUTM_DF[:3]]].tail(20))

    return CSUTM_DF





if __name__ == '__main__':
    pass










