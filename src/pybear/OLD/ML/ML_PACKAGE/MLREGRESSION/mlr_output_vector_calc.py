import sys, inspect, time
from debug import get_module_name as gmn
import numpy as np
import sparse_dict as sd
from ML_PACKAGE._data_validation import list_dict_validater as ldv
from general_data_ops import get_shape as gs




def mlr_output_vector_calc(DATA, data_run_orientation, COEFFS):

    this_module = gmn.get_module_name(str(sys.modules[__name__]))
    fxn = inspect.stack()[0][3]

    # GET INCOMING DATA TYPE, LIST OR SPARSE DICT
    data_format, DATA = ldv.list_dict_validater(DATA, 'DATA')

    # data_run_orientation SHOULD HAVE BEEN VALIDATED LONG BEFORE HERE

    coeff_format, COEFFS = ldv.list_dict_validater(COEFFS, 'COEFFS')
    if coeff_format=='SPARSE_DICT': raise Exception(f'\n*** {this_module}() >>> COEFFS ARE GOING INTO {fxn}() AS SPARSE_DICTS ***\n')
    else: COEFFS = COEFFS.reshape((1,-1))[0]
    del coeff_format

    data_cols = gs.get_shape('DATA', DATA, data_run_orientation)[1]

    if data_cols != len(COEFFS):
        print(f'\n*** {this_module}() >>> UNABLE TO COMPUTE OUTPUT VECTOR, COLUMN MISMATCH BETWEEN DATA ({data_cols}) '
              f'AND COEFFICIENT VECTOR ({len(COEFFS)}) ***\n')
        OUTPUT_VECTOR = []
    else:

        KWARGS = {'return_as': 'ARRAY', 'return_orientation': 'COLUMN'}

        if data_run_orientation=='ROW':
            if data_format=='ARRAY':
                OUTPUT_VECTOR = np.matmul(DATA.astype(np.float64), COEFFS.transpose().astype(np.float64),dtype=np.float64).reshape((1, -1))[0]
            elif data_format=='SPARSE_DICT':
                OUTPUT_VECTOR = sd.core_hybrid_matmul(DATA, COEFFS.transpose(), **KWARGS).reshape((1, -1))[0]

        if data_run_orientation=='COLUMN':
            # AVOID TRANSPOSING BIG DATA OBJECTS
            # USE (Ax)_T = (x_T)(A_T)
            if data_format == 'ARRAY':
                OUTPUT_VECTOR = np.matmul(COEFFS.astype(np.float64), DATA.astype(np.float64), dtype=np.float64).reshape((1, -1))[0]
            elif data_format == 'SPARSE_DICT':
                OUTPUT_VECTOR = sd.core_hybrid_matmul(COEFFS, DATA, **KWARGS).reshape((1, -1))[0]

        del KWARGS

    del data_format, data_cols, this_module

    return OUTPUT_VECTOR



























if __name__ == '__main__':
    from general_sound import winlinsound as wls

    # MODULE & TEST ARE GOOD AS OF 5/19/23

    rows = 100
    ctr = 0
    for columns in [50,1]:
        COEFFS = np.random.uniform(0, 1, columns)
        BASE_DATA = np.random.randint(0, 10, (columns, rows), dtype=np.int8)
        EXP_OUTPUT_VECTOR = np.matmul(BASE_DATA.transpose().astype(np.float64), COEFFS.astype(np.float64)).reshape((1,-1))[0]
        for data_orient in ['ROW', 'COLUMN']:
            for data_format in ['ARRAY', 'SPARSE_DICT']:
                for single_double in ['SINGLE', 'DOUBLE']:
                    ctr += 1
                    print(f'*' * 120)
                    print(f'\nRUNNING TRIAL {ctr}....')

                    if data_orient == 'ROW' or not columns==1: single_double = 'DOUBLE'    # ALWAYS MUST BE DOUBLE FOR THESE

                    desc = f'{single_double} {data_format}, {data_orient}, {columns} columns'

                    if data_orient == 'ROW': DATA = BASE_DATA.copy().transpose()
                    else: DATA = BASE_DATA.copy()
                    if data_format == 'SPARSE_DICT': DATA = sd.zip_list_as_py_float(DATA)
                    if single_double=='SINGLE': DATA = DATA[0]

                    ACT_OUTPUT_VECTOR = mlr_output_vector_calc(DATA, data_orient, COEFFS)

                    if not np.array_equiv(ACT_OUTPUT_VECTOR, EXP_OUTPUT_VECTOR):
                        print(f'\033[91m')
                        print(f'\n{desc}:')
                        print(f'\nACT_OUTPUT_VECTOR:')
                        print(ACT_OUTPUT_VECTOR)
                        print()
                        print(f'\nEXP_OUTPUT_VECTOR:')
                        print(EXP_OUTPUT_VECTOR)
                        print()
                        raise Exception(f'*** ACT OUTPUT VECTOR DOES NOT EQUAL EXP OUTPUT VECTOR ***')

                    print(f'*' * 120)

    print(f'\033[92m\n*** ALL TESTS PASSED ***\n')
    for _ in range(3): wls.winlinsound(888, 500); time.sleep(1)
