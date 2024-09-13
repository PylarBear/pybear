import sys, inspect
import numpy as np
import sparse_dict as sd
from debug import get_module_name as gmn
from ML_PACKAGE._data_validation import list_dict_validater as ldv
from MLObjects import MLObject as mlo


def mi_output_vector_calc(DATA, data_run_orientation, WINNING_COLUMNS, COEFFS):

    DATA = ldv.list_dict_validater(DATA, 'DATA')[1]

    # data_run_validation SHOULD BE VALIDATED WAY BEFORE THIS

    COEFFS = ldv.list_dict_validater(COEFFS, 'COEFFS')[1].reshape((1, -1))[0]

    WINNING_COLUMNS = ldv.list_dict_validater(WINNING_COLUMNS, 'WINNING_COLUMNS')[1].reshape((1,-1))[0]

    if len(WINNING_COLUMNS) != len(COEFFS):
        print(f'\n*** UNABLE TO COMPUTE OUTPUT VECTOR, SIZE MISMATCH BETWEEN DATA AND COEFFICIENT VECTOR ***\n')
        OUTPUT_VECTOR = []

    # EXTRACT WINNING COLUMNS FROM DATA AS []=ROW FOR matmul
    WinnerClass = mlo.MLObject(
                                DATA,
                                data_run_orientation,
                                name='DATA',
                                return_orientation='AS_GIVEN',
                                return_format='AS_GIVEN',
                                bypass_validation=True,
                                calling_module=gmn.get_module_name(str(sys.modules[__name__])),
                                calling_fxn=inspect.stack()[0][3]
    )

    WINNING_DATA = WinnerClass.return_columns(WINNING_COLUMNS, return_orientation='ROW', return_format='AS_GIVEN')
    del WinnerClass

    # TRANSPOSE TO [] = ROWS FOR MATMUL
    if isinstance(WINNING_DATA, np.ndarray):
        OUTPUT_VECTOR = np.matmul(WINNING_DATA.astype(np.float64), COEFFS.astype(np.float64)).reshape((1,-1))[0].astype(np.float64)
    elif isinstance(WINNING_DATA, dict):
        OUTPUT_VECTOR = sd.core_hybrid_matmul(
            WINNING_DATA, COEFFS.transpose(), COEFFS, return_as='ARRAY', return_orientation='COLUMN'
        ).reshape((1,-1))[0].astype(np.float64)

    del WINNING_DATA

    return OUTPUT_VECTOR






if __name__ == '__main__':

    # MODULE & TEST CODE GOOD 5/15/23

    rows = 20
    columns = 15
    winning_columns = 5
    BASE_DATA = np.random.randint(0, 10, (columns, rows))
    BASE_TARGET = np.random.randint(0, 2, (1, rows))
    COEFFS = np.random.uniform(0, 1, (1, winning_columns))
    WINNING_COLUMNS = np.random.randint(0, columns, winning_columns)

    WINNING_BASE_DATA = BASE_DATA[WINNING_COLUMNS, ...]

    EXP_OUTPUT_VECTOR = np.matmul(WINNING_BASE_DATA.transpose().astype(np.float64),
                                  COEFFS.transpose().astype(np.float64)).astype(np.float64).reshape((1,-1))[0]

    MASTER_DATA_FORMAT = ['ARRAY', 'SPARSE_DICT']
    MASTER_DATA_ORIENT = ['ROW', 'COLUMN']

    total_trials = np.product(list(map(len, (MASTER_DATA_FORMAT, MASTER_DATA_ORIENT))))

    ctr = 0
    for data_format in MASTER_DATA_FORMAT:
        for data_run_orientation in MASTER_DATA_ORIENT:

                ctr += 1
                print(f'*' * 120)
                print(f'RUnning trial {ctr} of {total_trials}...')

                GIVEN_DATA = BASE_DATA.copy()
                if data_run_orientation=='ROW': GIVEN_DATA = GIVEN_DATA.transpose()
                if data_format=='SPARSE_DICT': GIVEN_DATA = sd.zip_list_as_py_int(GIVEN_DATA)

                ACT_OUTPUT_VECTOR = mi_output_vector_calc(GIVEN_DATA, data_run_orientation, WINNING_COLUMNS, COEFFS)

                if not np.array_equiv(ACT_OUTPUT_VECTOR, EXP_OUTPUT_VECTOR):
                    print(f'\033[92m'); print(f'ACT_OUTPUT_VECTOR:')
                    print(ACT_OUTPUT_VECTOR)
                    print(f'\nEXP_OUTPUT_VECTOR:')
                    print(EXP_OUTPUT_VECTOR)
                    raise Exception(f'*** ACTUAL AND EXPECTED OUTPUT VECTOR ARE NOT EQUAL ***')

                print(f'*' * 120)

    print(f'\n\033[92m *** ALL TESTS PASSED *** \033[0m')





