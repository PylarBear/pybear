import numpy as n, pandas as p
import sparse_dict as sd
from pybear.data_validation import validate_user_input as vui
import time

# 12/18/22 THIS MODULE APPEARS TO TEST sd.sparse_transpose_from_list VS np.transpose AND sd.sparse_transpose()


start_rows = vui.validate_user_int(f'\nEnter start rows / cols > ', min=1)
final_rows = vui.validate_user_int(f'Enter final rows / cols > ', min=start_rows)
# cols = vui.validate_user_int(f'Enter final cols > ')
size_itvl = vui.validate_user_int(f'Enter size interval > ', min=1, max=final_rows-start_rows)

# start_sparsity = vui.validate_user_int(f'\nEnter start sparsity > ', min=0)
# final_sparsity = vui.validate_user_int(f'Enter final sparsity > ', min=start_sparsity, max=100)
# interval = vui.validate_user_int(f'Enter sparsity interval > ', min=1, max=final_sparsity-start_sparsity)
sparsity = 70

ROWS = list(range(start_rows,final_rows+1,size_itvl))
# SPARSITIES = list(range(start_sparsity,final_sparsity+1,interval))


ORIGINAL_TIMES = []
TEST_TIMES = []
RATIOS = []


for rows in range(start_rows,final_rows+1,size_itvl):

    RATIOS.append([])

    print(f'\nRunning {rows} x {rows}...')

    ORIGINAL_TIMES.clear()
    TEST_TIMES.clear()

    SPARSEDICT1 = sd.create_random(rows, rows, sparsity)
    LIST_DATA = n.array(sd.unzip_to_list(SPARSEDICT1)[0], dtype=object)

    print(f'TEST LIST_DATA')
    print(LIST_DATA)
    print()

    for _ in range(5):
        #######################################################
        t0 = time.time()

        # DUM1 = mt.matrix_transpose(LIST_DATA)
        # DUM1 = n.transpose(LIST_DATA)
        DUM1 = sd.sparse_transpose(SPARSEDICT1)

        tf = time.time()
        delta = tf - t0
        print(delta)
        ORIGINAL_TIMES.append(delta)
        #######################################################

        #######################################################
        t0 = time.time()

        # DUM2 = n.transpose(LIST_DATA)
        DUM2 = sd.sparse_transpose_from_list(LIST_DATA)

        tf = time.time()
        delta = tf - t0
        print(delta)
        TEST_TIMES.append(delta)
        #######################################################

    if not n.array_equiv(DUM1, DUM2):
        raise AssertionError(f'TRANSPOSES NOT EQUAL.')

    RATIOS[-1].append(n.average(TEST_TIMES) / n.average(ORIGINAL_TIMES))

    print()
    print(p.DataFrame(RATIOS, columns=['RATIO'], index=ROWS[:len(RATIOS)]))

print(f'\nFINAL PRINTOUT - AVERAGE TIME RATIO (TEST / ORIGINAL):\n')
DF = p.DataFrame(RATIOS, columns=['RATIO'], index=ROWS)

print(DF)
print()



