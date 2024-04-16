import numpy as n
from debug import time_memory_tester as tmt
import sparse_dict as sd
from general_data_ops import create_random_sparse_numpy as crsn


# UNZIP XTX FROM DICT IS ALWAYS FASTER THAN BUILD XTX FROM DATA AS ARRAY
##### TIME TEST 12/5/22, COMPARE SPEED FOR GETTING XTX AS ARRAY FROM MATMUL OF X AS ARRAY OR EXTRACT FROM A SD XTX #######
# (50, 5000) RESULTS =
# n.matmul                                          average, sdev: time = 3.300 sec, 0.010; mem = 95.000, 0.000
# sd.unzip_to_ndarray_float64                       average, sdev: time = 2.495 sec, 0.005; mem = 190.000, 0.000

# (500, 500) RESULTS =
# n.matmul                                          average, sdev: time = 0.449 sec, 0.001; mem = 0.000, 0.000
# sd.unzip_to_ndarray_float64                       average, sdev: time = 0.101 sec, 0.003; mem = 2.000, 0.000
#

# (5000, 50) RESULTS =
# n.matmul                                          average, sdev: time = 0.069 sec, 0.010; mem = 0.000, 0.000
# sd.unzip_to_ndarray_float64                       average, sdev: time = 0.021 sec, 0.007; mem = 0.000, 0.000
#







# _len_outer = 100
# _len_inner = 100000
# sparsity = 90
#
# DICT1 = sd.create_random_float(_len_outer, _len_inner, sparsity)
# DICT2 = sd.sparse_transpose(DICT1)
# DICT2_T = deepcopy(DICT1)

# DICT1_NP = sd.unzip_to_ndarray_float64(DICT1)[0]
# DICT2_NP = sd.unzip_to_ndarray_float64(DICT2)[0]
# DICT2_T_NP = sd.unzip_to_ndarray_float64(DICT2_T)[0]

for _len_outer, _len_inner in [(50, 5000), (500,500), (5000,50)]:

    _sparsity = 90

    NP1 = crsn.create_random_sparse_numpy(0, 10, (_len_outer, _len_inner), _sparsity, n.int32)
    SD1 = sd.zip_list_as_py_float(n.matmul(NP1.transpose(), NP1))

    print(f'RUNNING ({_len_outer}, {_len_inner})...')

    tmt.time_memory_tester(
                            ('n.matmul', n.matmul, [NP1.transpose(), NP1], {}), #{'DICT2_TRANSPOSE': DICT2_T, 'return_as':'ARRAY'},
                            ('sd.unzip_to_ndarray_float64', sd.unzip_to_ndarray_float64, [SD1], {}),  # {'DICT2_TRANSPOSE': DICT2_T, 'return_as':'ARRAY'},
                           )

    print(f'\n({_len_outer}, {_len_inner}) RESULTS = ')



