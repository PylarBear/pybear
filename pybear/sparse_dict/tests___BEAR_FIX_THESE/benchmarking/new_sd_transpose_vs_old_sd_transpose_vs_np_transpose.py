import numpy as np
import sparse_dict as sd
from debug import time_memory_tester as tmt
from general_data_ops import create_random_sparse_numpy as crsn


# THIS MODULE COMPARES THE SPEED OF OLD (CURRENTLY USED) core_sparse_transpose(), new_sparse_transpose,
# and np.transpose() AS BASELINE




ctr = 0
for _rows, _columns in ((100,5000), (5000,100), (707,707)):
    ctr += 1
    print(f'Running trial ({_rows}, {_columns})...')
    # CREATE OBJECTS
    print(f'\nCreating objects....')

    # SIMULATE A BINARY OBJECT W []=ROWS
    BASE_NP_OBJECT = crsn.create_random_sparse_numpy(0,100,(_rows,_columns), 50, np.float64)
    TEST_SD = sd.zip_list_as_py_int(BASE_NP_OBJECT)

    print(f'Done.\n')

    tmt.time_memory_tester(
                            (f'core_sparse_transpose ({_rows}, {_columns})', sd.core_sparse_transpose, [TEST_SD], {}),
                            (f'new_sparse_transpose ({_rows}, {_columns})', sd.new_sparse_transpose, [TEST_SD], {}),
                            (f'np_transpose ({_rows}, {_columns})', np.transpose, [BASE_NP_OBJECT], {}),
                             number_of_trials=10,
                             rest_time=3
    )


print(f'\n\033[92m*** TESTS COMPLETE ***\n')
















