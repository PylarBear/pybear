import time
import numpy as np
import sparse_dict as sd
from ML_PACKAGE._data_validation import list_dict_validater as ldv


def mlr_error_calc(OUTPUT_VECTOR, TARGET_VECTOR):

    output_format, OUTPUT_VECTOR = ldv.list_dict_validater(OUTPUT_VECTOR, 'OUTPUT_VECTOR')
    target_format, TARGET_VECTOR = ldv.list_dict_validater(TARGET_VECTOR, 'TARGET_VECTOR')

    if output_format == 'SPARSE_DICT': OUTPUT_VECTOR = sd.unzip_to_ndarray(OUTPUT_VECTOR)[0]
    if target_format == 'SPARSE_DICT': TARGET_VECTOR = sd.unzip_to_ndarray(TARGET_VECTOR)[0]

    del output_format, target_format

    sse = np.sum((TARGET_VECTOR.reshape((1, -1))[0] - OUTPUT_VECTOR.reshape((1, -1))[0]) ** 2)

    return sse
















if __name__ == '__main__':

    # MODULE AND TEST GOOD 5/19/23

    from general_sound import winlinsound as wls

    rows = 5
    BASE_OUTPUT = np.random.uniform(0, 10, (1, rows))
    BASE_TARGET = np.random.uniform(0, 1, (1, rows))

    exp_error = np.sum(np.power(BASE_OUTPUT - BASE_TARGET, 2)).astype(np.float64)

    MASTER_OUTPUT_FORMAT = ['ARRAY', 'SPARSE_DICT']
    MASTER_OUTPUT_ORIENT = ['ROW', 'COLUMN']
    MASTER_TARGET_FORMAT = ['ARRAY', 'SPARSE_DICT']
    MASTER_TARGET_ORIENT = ['ROW', 'COLUMN']

    total_trials = np.product(list(map(len,
                                       (MASTER_OUTPUT_FORMAT, MASTER_OUTPUT_ORIENT, MASTER_TARGET_FORMAT,
                                        MASTER_TARGET_ORIENT))))

    ctr = 0
    for output_format in MASTER_OUTPUT_FORMAT:
        for output_orient in MASTER_OUTPUT_ORIENT:
            for target_format in MASTER_TARGET_FORMAT:
                for target_orient in MASTER_TARGET_ORIENT:
                    ctr += 1
                    print(f'*' * 120)
                    print(f'RUnning trial {ctr} of {total_trials}...')

                    GIVEN_OUTPUT = BASE_OUTPUT.copy()
                    if output_orient == 'ROW': GIVEN_OUTPUT = GIVEN_OUTPUT.transpose()
                    if output_format == 'SPARSE_DICT': GIVEN_OUTPUT = sd.zip_list_as_py_float(GIVEN_OUTPUT)

                    GIVEN_TARGET = BASE_TARGET.copy()
                    if target_orient == 'ROW': GIVEN_TARGET = GIVEN_TARGET.transpose()
                    if target_format == 'SPARSE_DICT': GIVEN_TARGET = sd.zip_list_as_py_float(GIVEN_TARGET)

                    act_error = mlr_error_calc(GIVEN_OUTPUT, GIVEN_TARGET)

                    if not act_error == exp_error:
                        print(f'\033[91m')

                        print(f'BASE_OUTPUT:')
                        print(BASE_OUTPUT)
                        print()
                        print(f'BASE_TARGET:')
                        print(BASE_TARGET)
                        print()
                        print(f'output_format = {output_format}')
                        print(f'output_orient = {output_orient}')
                        print(f'target_format = {target_format}')
                        print(f'target_orient = {target_orient}')
                        print(f'act_error = {act_error}, exp_error = {exp_error}')
                        time.sleep(0.5)
                        raise Exception(f'*** ACTUAL AND EXPECTED error ARE NOT EQUAL ***')

                    print(f'*' * 120)

    print(f'\033[92m *** ALL TESTS PASSED *** \033[0m')
    for _ in range(3): wls.winlinsound(888, 500); time.sleep(1)


