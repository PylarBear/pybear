import numpy as np
import random
from ML_PACKAGE.NN_PACKAGE.gd_run import error_calc as ec, output_vector_calc as ovc


# 2-2-22 DONT PUT THIS INSIDE NNRun, KEEP IT SEPARATE, THINKING THIS ALGORITM WILL BE NEEDED FOR SVM

# CALLED BY NNCoreRunCode
def cd_engine(method, ARRAY_OF_NODES, LIST_OF_ELEMENTS, SELECT_LINK_FXN, NEW_TARGET_VECTOR, BATCH_MATRIX, OUTPUT_VECTOR,
              LEARNING_RATE, cost_fxn, non_neg_coeffs, new_error_start, rglztn_type, rglztn_fctr, activation_constant,
              iteration):

    # BEAR 24_04_09_16_03_00 THIS WAS ONCE IN ITS OWN MODULE BUT WAS ONLY CALLED HERE
    # MELD INTO THE ADJOINING LANDSCAPE AS DESIRED
    def randomize_elements(ELEMENT_LIST_WIP, RANDOMIZED_ELEMENT_LIST):
        ELEMENT_WIP2 = ELEMENT_LIST_WIP.copy()
        RANDOMIZED_ELEMENT_LIST.clear()
        while len(ELEMENT_WIP2) > 0:
            index = random.choice(range(len(ELEMENT_WIP2)))
            RANDOMIZED_ELEMENT_LIST.append(ELEMENT_WIP2.pop(index))

        return RANDOMIZED_ELEMENT_LIST


    for nle in randomize_elements(LIST_OF_ELEMENTS, []):  # nle  = NODE, LIST_OBJECT, ELEMENT

        # GET JIGGLE
        jiggle = LEARNING_RATE[nle[0]][iteration]

        # RETAIN INITIAL STATE
        NEW_ERROR_MID = new_error_start
        orig_element = ARRAY_OF_NODES[nle[0]][nle[1]][nle[2]]

        # CALCULATE -JIGGLE*****************************************************************
        if non_neg_coeffs == 'N': lo_element = orig_element - jiggle
        elif non_neg_coeffs == 'Y': lo_element = max(orig_element - jiggle, 0)
        ARRAY_OF_NODES[nle[0]][nle[1]][nle[2]] = lo_element

        OUTPUT_VECTOR = ovc.output_vector_calc(BATCH_MATRIX, ARRAY_OF_NODES, SELECT_LINK_FXN, OUTPUT_VECTOR, activation_constant)
        NEW_ERROR_LOW = ec.error_calc(ARRAY_OF_NODES, NEW_TARGET_VECTOR, OUTPUT_VECTOR, cost_fxn, new_error_start,
                                      SELECT_LINK_FXN, rglztn_type, rglztn_fctr)
        # ***********************************************************************************

        # CALCULATE +JIGGLE******************************************************************
        ARRAY_OF_NODES[nle[0]][nle[1]][nle[2]] = orig_element + jiggle

        OUTPUT_VECTOR = ovc.output_vector_calc(BATCH_MATRIX, ARRAY_OF_NODES, SELECT_LINK_FXN, OUTPUT_VECTOR, activation_constant)
        NEW_ERROR_HIGH = ec.error_calc(ARRAY_OF_NODES, NEW_TARGET_VECTOR, OUTPUT_VECTOR, cost_fxn, new_error_start,
                                       SELECT_LINK_FXN, rglztn_type, rglztn_fctr)
        # ************************************************************************************

        # PICK BEST OUT OF -JIGGLE, ORIGINAL, +JIGGLE
        if method.upper() == 'MIN':
            if NEW_ERROR_LOW < min(NEW_ERROR_MID, NEW_ERROR_HIGH): _ = lo_element
            elif NEW_ERROR_MID <= min(NEW_ERROR_LOW, NEW_ERROR_HIGH): _ = orig_element
            elif NEW_ERROR_HIGH < min(NEW_ERROR_LOW, NEW_ERROR_MID): _ = orig_element + jiggle
            else: raise AssertionError(f'cd_engine(), MIN METHOD, OPTIMIZATION LOGIC HAS FAILED.')

        elif method.upper() == 'MAX':
            if NEW_ERROR_LOW > max(NEW_ERROR_MID, NEW_ERROR_HIGH): _ = lo_element
            elif NEW_ERROR_MID >= max(NEW_ERROR_LOW, NEW_ERROR_HIGH): _ = orig_element
            elif NEW_ERROR_HIGH > max(NEW_ERROR_LOW, NEW_ERROR_MID): _ = orig_element + jiggle
            else: raise AssertionError(f'cd_engine(), MAX METHOD, OPTIMIZATION LOGIC HAS FAILED.')

        else:
            raise ValueError(f'\n*** INVALID method IN cd_engine.  MUST BE "MIN" OR "MAX". ***\n')

        ARRAY_OF_NODES[nle[0]][nle[1]][nle[2]] = _

    # IF LIST OF ELEMENTS IS EMPTY, THE for LOOP ABOVE IS BYPASSED AND BELOW BLOWS UP BECAUSE "NEW_" HAVENT BEEN DECLARED
    try: new_error_start = np.min([NEW_ERROR_MID, NEW_ERROR_LOW, NEW_ERROR_HIGH])
    except: pass

    return ARRAY_OF_NODES, OUTPUT_VECTOR, new_error_start

