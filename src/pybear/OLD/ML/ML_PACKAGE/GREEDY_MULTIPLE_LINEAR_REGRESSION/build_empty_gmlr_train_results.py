import numpy as np
import pandas as pd
from ML_PACKAGE._data_validation import list_dict_validater as ldv

# THIS MAKES ALL DATAFRAME HEADERS AND INDEXES "UNSPARSE"
pd.set_option('display.multi_sparse', False, 'display.colheader_justify', 'center')
pd.set_option('display.max_columns', None, 'display.width', 150, 'display.max_colwidth', 35)
pd.options.display.float_format = '{:,.5f}'.format


def build_empty_gmlr_train_results(HEADER):

    HEADER = ldv.list_dict_validater(HEADER, 'HEADER')[1][0]     # GET AS [], NOT [[]]


    TRAIN_RESULTS_HEADER = pd.MultiIndex(
        levels=[['COLUMN', 'INDIV', 'FINAL', 'CUMUL'],
                ['NAME', 'R', 'R2', 'ADJ R2', 'F', 'COEFFS', 'p VALUE']],
        codes=[[1, 1, 1, 1, 2, 2, 3, 3, 3, 3], [1, 2, 3, 4, 5, 6, 1, 2, 3, 4]])

    # OLD
    # TRAIN_RESULTS_HEADER = [
    #     ['COLUMN', ' INDIV  ', ' INDIV  ', ' INDIV ', ' INDIV  ', 'FINAL ', ' FINAL ', 'CUMUL', 'CUMUL', ' CUMUL ', 'CUMUL'],
    #     [' NAME ', '   R    ', '   R2   ', ' ADJ R2', '   F    ', 'COEFFS', 'p VALUE', '  R  ', ' RSQ ', ' ADJ R2', '  F  ']
    #     ]


    # AS OF 5/23 'COLUMN NAME' IS IN INDEX, HEADER IS
    # TRAIN_RESULTS_HEADER = [
    #     [' INDIV  ', ' INDIV  ', ' INDIV ', ' INDIV  ', 'FINAL ', ' FINAL ', 'CUMUL', 'CUMUL', ' CUMUL ', 'CUMUL'],
    #     ['   R    ', '   R2   ', ' ADJ R2', '   F    ', 'COEFFS', 'p VALUE', '  R  ', ' RSQ ', ' ADJ R2', '  F  ']
    #     ]

    EMPTY_RESULTS = pd.DataFrame(index=HEADER, columns=TRAIN_RESULTS_HEADER, dtype=object).fillna('-')

    # DIMENSIONS ARE len(HEADER) ROWS x len(TRAIN_RESULTS_HEADER) COLUMNS

    return EMPTY_RESULTS






if __name__ == '__main__':

    from MLObjects.TestObjectCreators import test_header as th


    TEST_HEADER = th.test_header(20000)[..., -20:]

    TEST_RESULTS = build_empty_gmlr_train_results(TEST_HEADER)

    print(TEST_RESULTS)

    if not np.array_equiv(TEST_HEADER, TEST_RESULTS.index):
        raise Exception(f'*** DF INDEX IS NOT EQUAL TO GIVEN HEADER ***')
    else:
        print(f'\033[92m *** TEST PASSED ***\033[0m')

























