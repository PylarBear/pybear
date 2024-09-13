import numpy as np
import sparse_dict as sd
from utilities._benchmarking import time_memory_benchmark as tmb
from new_numpy._random_ import sparse
from sparse_dict._transform import zip_array, unzip_to_ndarray
from sparse_dict._linalg import core_sparse_transpose



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
        return sd.unzip_to_ndarray(sd.sparse_transpose(SD_OBJECT), dtype=np.float64).transpose()
    else:
        return sd.unzip_to_ndarray(SD_OBJECT, dtype=np.float64)


def conditional_zip_list_as_py_float(NP_OBJECT, rows, cols):

    if rows >= 1000*cols:
        return sd.sparse_transpose(zip_array(NP_OBJECT.transpose(), dtype=float))
    else:
        return sd.zip_array(NP_OBJECT, dtype=float)



for _rows, _columns in ((1000,1), (2000,1), (4000,1), (5000, 1), (10000, 1), (20000,1), (30000,1)):

    # CREATE OBJECTS
    print(f'\nCreating objects....')

    # SIMULATE A BINARY OBJECT W []=ROWS
    BASELINE_OBJECT = sparse(0,2,(_rows,_columns), 50, dtype=np.uint8)

    DICT_TO_BE_UNZIPPED = zip_array(BASELINE_OBJECT, dtype=float)
    LIST_TO_BE_ZIPPED = BASELINE_OBJECT.copy()
    DICT_TO_BE_TRANSPOSED = zip_array(BASELINE_OBJECT.transpose(), dtype=float)  # TRANSPOSE TO OPPOSITE OF BASELINE SO IT COMES OUT ==BASELINE

    print(f'Done.\n')

    RESULTS = \
        tmb(
            (f'unzip_to_ndarray_float64 ({_rows},{_columns})',
            unzip_to_ndarray,
            [DICT_TO_BE_UNZIPPED],
            {'dtype': np.float64}
            ),
            (f'conditional_unzip_to_ndarray_float64 ({_rows}, {_columns})',
             conditional_unzip_to_ndarray_float64,
             [DICT_TO_BE_UNZIPPED, _rows, _columns],
             {}
             ),
            (f'zip_list_as_py_float ({_rows}, {_columns})',
            zip_array,
            [LIST_TO_BE_ZIPPED],
            {'dtype': float}
            ),
            (f'conditional_zip_list_as_py_float ({_rows}, {_columns})',
             conditional_zip_list_as_py_float,
             [LIST_TO_BE_ZIPPED, _rows, _columns],
             {}
             ),
            (f'sparse_transpose ({_rows}, {_columns})',
            core_sparse_transpose,
            [DICT_TO_BE_TRANSPOSED],
            {}
            ),
            number_of_trials=5,
            rest_time=2
        )







