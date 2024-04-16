import numpy as np, time
from copy import deepcopy
from debug import time_memory_tester as tmt
import sparse_dict as sd
from general_data_ops import create_random_sparse_numpy as crsn



##### TIME TESTS 12/7/22, COMPARE SPEED FOR GETTING XTX AS ARRAY FROM MATMUL OF X AS ARRAY OR linalg.inv(XTX_INV) #######



#### BEAR VERIFY THAT NP MATMUL REALLY TAKES FOREVER ####
TEST_NP = crsn.create_random_sparse_numpy(0,10,(2000,1000),90, np.int32)
NP_T = TEST_NP.transpose()
TIMES = np.empty((1,5),dtype=np.float64)[0]
for _ in range(5):
    t0 = time.time()
    DUM = np.matmul(NP_T, TEST_NP)
    _time = time.time() - t0
    print(f'time = {_time} sec')
    TIMES[_] = _time
print(f'average time = {np.average(TIMES)} sec')
quit()

# IT REALLY IS THAT FLIPPIN SLOW :(
# time = 26.16 sec
# time = 27.45 sec
# time = 38.54 sec
# time = 34.14 sec
# time = 25.92 sec
# average time = 30.45 sec

## END BEAR VERIFY #####################################



##### TIME TEST 12/7/22, COMPARE SPEED FOR GETTING XTX AS ARRAY FROM MATMUL OF X AS ARRAY OR linalg.inv(XTX_INV) #######
# NOT EVEN CLOSE.  FOR NUMPY, linalg.inv BEATS matmul(XT,X) EASILY
# FOR SD, zip_list(linalg.inv(unzip(SD_XTX))) EASILY BEATS core_matmul(XT, X)

# np.linalg_inv on XTX_INV                          average, sdev: time = 0.212 sec, 0.003; mem = 8.000, 0.000
# np.matmul on XT and X                             average, sdev: time = 26.648 sec, 0.967; mem = 4.000, 0.000
# sd.unzip --> np.inv --> sd.zip                    average, sdev: time = 0.948 sec, 0.065; mem = 88.333, 0.471
# sd.core_matmul                                    average, sdev: time = 35.395 sec, 1.441; mem = 65.000, 0.000



_len_outer, _len_inner = (2000,1000)
_sparsity = 90
print(f'Building test objects....')
NP1 = crsn.create_random_sparse_numpy(0, 10, (_len_outer, _len_inner), _sparsity, np.int32)
NP1_T = NP1.transpose()
NP_XTX = np.matmul(NP1_T, NP1)
NP_XTX_INV = np.linalg.inv(NP_XTX)
SD1 = sd.zip_list_as_py_float(NP1)
SD1_T = sd.sparse_transpose(SD1)
SD_XTX = sd.core_matmul(SD1_T, SD1, DICT2_TRANSPOSE=SD1_T)
SD_XTX_INV = sd.zip_list_as_py_float(np.linalg.inv(sd.unzip_to_ndarray_float64(SD_XTX)[0]))
print(f'Done.\n')

def sd_xtx_inv(XTX_INV):
    return sd.zip_list_as_py_float(np.linalg.inv(sd.unzip_to_ndarray_float64(SD_XTX_INV)[0]))


print('*'*80)
print(f'\nNP RUNNING ({_len_outer}, {_len_inner})...')

print(f'\n({_len_outer}, {_len_inner}) RESULTS = ')
tmt.time_memory_tester(
    ('np.linalg_inv on XTX_INV', np.linalg.inv, [NP_XTX_INV], {}), #{'DICT2_TRANSPOSE': DICT2_T, 'return_as':'ARRAY'},
    ('np.matmul on XT and X', np.matmul, [NP1_T, NP1], {})  # {'DICT2_TRANSPOSE': DICT2_T, 'return_as':'ARRAY'},
)


print(f'\nSD ({_len_outer}, {_len_inner}) RESULTS = ')
tmt.time_memory_tester(
    ('sd.unzip --> np.inv --> sd.zip', sd_xtx_inv, [SD_XTX_INV], {}), #{'DICT2_TRANSPOSE': DICT2_T, 'return_as':'ARRAY'},
    ('sd.core_matmul', sd.core_matmul, [SD1_T, SD1], {'DICT2_TRANSPOSE': SD1_T, 'return_as': 'SPARSE_DICT'})
)











