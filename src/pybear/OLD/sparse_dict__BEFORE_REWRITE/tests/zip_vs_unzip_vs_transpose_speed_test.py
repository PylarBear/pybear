import numpy as np
import sparse_dict as sd
from debug import time_memory_tester as tmt
from general_data_ops import create_random_sparse_numpy as crsn


# THIS MODULE COMPARES THE SPEED OF ZIPPING A LIST TO SD, UNZIPPING A LIST TO NP, AND TRANSPOSING AN SD
# ALL THE SAME OBJECT IN DIFFERENT FORMATS
# MAKE STARTING OBJECTS SO THAT THEY RETURN THE EXACT SAME THING


# TIME TESTS 12/8/22
# AS 1000000x1 OBJECT OF 0,1 E.G. [[0][1][1].......]
# unzip_to_ndarray_float64    average, sdev: time = 15.25 sec, 0.129; mem = 7.333, 0.943
# zip_list_as_py_float        average, sdev: time = 48.56 sec, 0.149; mem = 317.000, 1.414
# sparse_transpose            average, sdev: time = 1.157 sec, 0.010; mem = 285.000, 0.816  (TRANSPOSED TO [[0],[1],....]])

# AS 1x1000000 OBJECT OF 0,1 E.G. [[0,1,0.......]]
# unzip_to_ndarray_float64    average, sdev: time = 0.201 sec, 0.004; mem = 8.000, 0.000
# zip_list_as_py_float        average, sdev: time = 0.333 sec, 0.005; mem = 66.000, 0.000
# sparse_transpose            average, sdev: time = 2.756 sec, 0.030; mem = 40.000, 0.000  (TRANSPOSED TO [[0,1,0,....]])

# AS 100x10000 OBJECTS OF 0,1 E.G. 100x[[0,1,0,....]]
# unzip_to_ndarray_float64    average, sdev: time = 0.159 sec, 0.006
# zip_list_as_py_float        average, sdev: time = 0.289 sec, 0.009
# sparse_transpose            average, sdev: time = 0.437 sec, 0.007  (TRANSPOSED TO 100x[[0,1,0,....]])

# AS 10000x100 OBJECTS OF 0,1 E.G. 10000x[[0,1,0,....]]
# unzip_to_ndarray_float64    average, sdev: time = 0.312 sec, 0.005
# zip_list_as_py_float        average, sdev: time = 0.780 sec, 0.007
# sparse_transpose            average, sdev: time = 0.522 sec, 0.005  (TRANSPOSED TO 10000x[[],[],....])



def conditional_unzip_to_ndarray_float64(SD_OBJECT, rows, cols):

    if rows >= 2000*cols:
        return sd.unzip_to_ndarray_float64(sd.sparse_transpose(SD_OBJECT))[0].transpose()
    else:
        return sd.unzip_to_ndarray_float64(SD_OBJECT)

def conditional_zip_list_as_py_float(NP_OBJECT, rows, cols):

    if rows >= 1000*cols:
        return sd.sparse_transpose(sd.zip_list_as_py_float(NP_OBJECT.transpose()))
    else:
        return sd.zip_list_as_py_float(NP_OBJECT)



for _rows, _columns in ((1000,1), (2000,1), (4000,1), (5000, 1), (10000, 1), (20000,1), (30000,1)):

    # CREATE OBJECTS
    print(f'\nCreating objects....')

    # SIMULATE A BINARY OBJECT W []=ROWS
    BASELINE_OBJECT = crsn.create_random_sparse_numpy(0,2,(_rows,_columns), 50, np.float64)

    DICT_TO_BE_UNZIPPED = sd.zip_list_as_py_float(BASELINE_OBJECT)
    LIST_TO_BE_ZIPPED = BASELINE_OBJECT
    DICT_TO_BE_TRANSPOSED = sd.zip_list_as_py_float(BASELINE_OBJECT.transpose())  # TRANSPOSE TO OPPOSITE OF BASELINE SO IT COMES OUT ==BASELINE

    print(f'Done.\n')

    tmt.time_memory_tester(
                            (f'unzip_to_ndarray_float64 ({_rows},{_columns})',
                            sd.unzip_to_ndarray_float64,
                            [DICT_TO_BE_UNZIPPED],
                            {}
                            ),
                            (f'zip_list_as_py_float ({_rows}, {_columns})',
                            sd.zip_list_as_py_float,
                            [LIST_TO_BE_ZIPPED],
                            {}
                            ),
                            (f'sparse_transpose ({_rows}, {_columns})',
                            sd.sparse_transpose,
                            [DICT_TO_BE_TRANSPOSED],
                            {}
                            ),
                            number_of_trials=5,
                            rest_time=2
    )


    tmt.time_memory_tester(
                            (f'conditional_unzip_to_ndarray_float64 ({_rows}, {_columns})',
                            conditional_unzip_to_ndarray_float64,
                            [DICT_TO_BE_UNZIPPED, _rows, _columns],
                            {}
                            ),
                            (f'conditional_zip_list_as_py_float ({_rows}, {_columns})',
                            conditional_zip_list_as_py_float,
                            [LIST_TO_BE_ZIPPED, _rows, _columns],
                            {}
                            ),
                            (f'sparse_transpose ({_rows}, {_columns})',
                            sd.sparse_transpose,
                            [DICT_TO_BE_TRANSPOSED],
                            {}
                            ),
                            number_of_trials=5,
                            rest_time=2
    )


