import numpy as n, pandas as p
import sparse_dict as sd
from pybear.data_validation import validate_user_input as vui
import time, winsound

# THIS MODULE CALCULATES THE RATIO OF TIMES FOR sparse_ATA(X) / np.matmul(X, X.T) FOR VARIOUS X SIZES AND SPARSITIES

start_rows = vui.validate_user_int(f'\nEnter start rows > ', min=1)
final_rows = vui.validate_user_int(f'Enter final rows > ', min=start_rows)
# cols = vui.validate_user_int(f'Enter final cols > ')
size_itvl = vui.validate_user_int(f'Enter size interval > ', min=1, max=final_rows-start_rows)
num_cols = vui.validate_user_int(f'Enter number of columns > ', min=1, max=start_rows)
start_sparsity = vui.validate_user_int(f'\nEnter start sparsity > ', min=0)
final_sparsity = vui.validate_user_int(f'Enter final sparsity > ', min=start_sparsity, max=100)
interval = vui.validate_user_int(f'Enter sparsity interval > ', min=1, max=final_sparsity-start_sparsity)

ROWS = list(range(start_rows,final_rows+1,size_itvl))
SPARSITIES = list(range(start_sparsity,final_sparsity+1,interval))

TIMES = []

MATMUL_TIMES = []
CORE_SYM_MATMUL = []
SPARSE_ATA_TIMES = []


for rows in range(start_rows,final_rows+1,size_itvl):

    TIMES.append([])

    for sparsity in range(start_sparsity, final_sparsity+1, interval):

        print(f'\nRunning {rows} x {num_cols}...')#, sparsity = {sparsity}...')

        MATMUL_TIMES.clear()
        CORE_SYM_MATMUL.clear()
        SPARSE_ATA_TIMES.clear()

        SPARSEDICT1 = sd.create_random(rows, num_cols, sparsity)
        # SPARSEDICT1 = sd.create_random(rows, rows, sparsity)
        # LIST_DATA = sd.unzip_to_list(SPARSEDICT1)[0]
        # from linear_algebra import matrix_transpose as mt
        # LIST_DATA_T = mt.matrix_transpose(LIST_DATA)
        LIST_DATA = sd.unzip_to_ndarray(SPARSEDICT1)[0]
        LIST_DATA_T = LIST_DATA.transpose()



        for _ in range(5):
            # NP MATMUL ####################################################################################################
            t0 = time.time()
            DUM1 = n.matmul(LIST_DATA_T, LIST_DATA, dtype=float)
            MATMUL_TIMES.append(time.time() - t0)
            print(f'MATMUL TIME = {time.time() - t0}')
            # END NP MATMUL ###############################################################################################
            # CORE SYM MATMUL ####################################################################################
            t0 = time.time()
            SPARSEDICT2 = sd.zip_list(LIST_DATA_T)
            DUM2 = sd.core_symmetric_matmul(SPARSEDICT2, SPARSEDICT1, DICT2_TRANSPOSE=SPARSEDICT2, return_as=None)
            CORE_SYM_MATMUL.append(time.time() - t0)
            print(f'CORE_SYM_MATMUL TIME = {time.time() - t0}')
            # END OLD CORE SYM MATMUL ####################################################################################
            # # SPARSE_ATA ####################################################################################
            # t0 = time.time()
            # DUM = sd.sparse_ATA(SPARSEDICT1, DICT1_TRANSPOSE=SPARSEDICT2, return_as=None)
            # SPARSE_ATA_TIMES.append(time.time() - t0)
            # print(f'SPARSE_ATA TIME = {time.time() - t0}')
            # # END OLD CORE SYM MATMUL ####################################################################################

        # if not n.array_equiv(n.array(DUM1), n.array(DUM2)):
        #     raise Exception(f'OUTPUTS ARE NOT EQUAL!')

        del DUM1, DUM2
        ''' 
        RATIO OF sd.core_sym_matmul TIMES TO np.matmul TESTS 9/20/22 -- SQUARE MATRIX, ROWS(COLS) ON LEFT, SPARSITY ON TOP
        LOOK LIKE USING intersecion ON set CLASS PULLS AWAY AS INPUT MATRIX GETS BIGGER
        
        set.intersection filling of mmult matrix
                   50          60          70          80          90
        100   496.458164           inf  308.581195  120.154719         inf
        200   840.133739    644.867415  447.500030  354.028215  131.297905
        300  1076.802550    653.786988  475.064799  279.906366  145.570253
        400  1215.335489    707.593673  474.149833  258.468006  121.058113
        500  1108.733663    925.051163  556.964245  259.061624  142.461903
        600   1332.413163   939.249740  599.685934  290.673046  150.833191
        700   1189.119986   962.986038  583.236971  357.491015  156.487191
        800   1457.887878  1178.979667  818.332212  440.410192  176.952364
        900   1524.025535  1121.239043  758.608534  363.860038  164.524824
        1000  1402.345779  1123.676874  737.454833  374.113226  165.388897
        
        key iterator filling of mmult matrix
                  50            60          70          80          90
        100    463.604912         inf  128.930444   90.577024         inf
        200    826.642184   509.466106  321.585538  497.761950  126.958246
        300    948.688384   878.084516  331.680668  265.281927  169.845089
        400   1332.614309   892.241164  453.416061  309.011392  131.660602
        500   1249.198028   920.406032  731.667926  318.397558  137.330728
        600   1584.032464  1129.856743   615.674256  485.681978  137.698824
        700   1469.816941   934.815686   766.835332  430.656091  186.758112
        800   1411.216114  1119.835771   672.460848  417.077681  173.851524
        900   1559.903943  1163.111235   853.534633  443.528728  195.207737
        1000  1781.196327  1348.274144  1111.852135  437.722877  206.882713
        '''
        # TIMES[-1].append(n.average(MATMUL_TIMES))
        # TIMES[-1].append(n.average(CORE_SYM_MATMUL))
        # TIMES[-1].append(n.average(SPARSE_ATA_TIMES))
        TIMES[-1].append(n.average(CORE_SYM_MATMUL) / n.average(MATMUL_TIMES))
        # TIMES[-1].append(n.average(SPARSE_ATA_TIMES) / n.average(MATMUL_TIMES))

        print()
        print(p.DataFrame(TIMES,
                          columns=[SPARSITIES[:len(TIMES[0])]],
                          index=ROWS[:len(TIMES)]))

# print(f'\nFINAL PRINTOUT - AVERAGE TIME RATIO (SPARSE / MATMUL):\n')
print()
DF = p.DataFrame(TIMES, columns=SPARSITIES, index=ROWS)

del MATMUL_TIMES, CORE_SYM_MATMUL, SPARSE_ATA_TIMES,SPARSEDICT1, LIST_DATA,LIST_DATA_T,SPARSEDICT2

print(DF)
print()
winsound.Beep(888, 1000)









