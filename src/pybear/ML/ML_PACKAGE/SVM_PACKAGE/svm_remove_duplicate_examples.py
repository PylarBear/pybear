import numpy as n
from copy import deepcopy
from linear_algebra import matrix_transpose as mt


def svm_remove_duplicate_examples(DATA, TARGET):
    # DATA MUST COME IN AS [ [] = COLUMNS ]
    # TARGET MUST COME IN AS [[]]

    # CLEAN OUT DUPLICATE EXAMPLES ###################################################################################
    print(f'Cleaning out duplicates examples...')
    print(f'Starting DATA length: {len(DATA[0])}')

    DATA_WIP = mt.matrix_transpose(DATA)                                # GO TO LISTS = EXAMPLES
    TARGET_WIP = mt.matrix_transpose(TARGET)                            # GO TO LISTS = EXAMPLES

    for example1_idx in range(len(DATA_WIP)-2 ,-1 ,-1):
        for example2_idx in range(len(DATA_WIP)-1, example1_idx, -1):
            # print(f'example1_idx = {example1_idx}')
            # print(f'example2_idx = {example2_idx}')
            if n.array_equiv(DATA_WIP[example2_idx], DATA_WIP[example1_idx]) and \
                    n.array_equiv(TARGET_WIP[0][example2_idx], TARGET_WIP[0][example1_idx]):
                DATA_WIP = DATA_WIP.pop(example2_idx)
                TARGET_WIP = TARGET_WIP.pop(example2_idx)
                print(f'DELETED ROW {example2_idx} FOR EQUALITY WITH ROW {example1_idx}')

    DATA_WIP = mt.matrix_transpose(DATA_WIP)
    TARGET_WIP = mt.matrix_transpose(TARGET_WIP)

    print(f'POST-CLEAN DATA: {len(DATA_WIP[0])} ROWS, {len(DATA_WIP)} COLUMNS')
    # END CLEAN OUT DUPLICATE EXAMPLES ###################################################################################
    ####################################################################################################################

    return DATA, TARGET



















