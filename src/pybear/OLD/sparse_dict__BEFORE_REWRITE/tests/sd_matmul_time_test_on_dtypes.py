import numpy as n, sparse_dict as sd
from copy import deepcopy

from debug import time_memory_tester as tmt

rows = 75000
cols = 10
sparsity = 90

ORIG_NP = n.random.randint(1,10,(cols,rows))

############################################################################################################
DICT1_N = {int(_): {int(___): float(____) for ___, ____ in enumerate(__)} for _,__ in enumerate(ORIG_NP)}

DICT2_N = {}
for old_inner_idx in range(rows):
    DICT2_N[int(old_inner_idx)] = {}
    for old_outer_idx in range(cols):
        DICT2_N[old_inner_idx][int(old_outer_idx)] = float(DICT1_N[old_outer_idx][old_inner_idx])

DICT2_N_T = DICT1_N

############################################################################################################

############################################################################################################
DICT1_O = sd.zip_list_as_float64(ORIG_NP)
DICT2_O = sd.zip_list_as_float64(ORIG_NP.transpose()) #sd.core_sparse_transpose(DICT1_O)
DICT2_O_T = DICT1_O


tmt.time_memory_tester(
    ('test_new_core_matmul_w_', sd.test_new_core_matmul, [deepcopy(DICT1_O), deepcopy(DICT2_O)], {'DICT2_TRANSPOSE': DICT2_O_T}),
    ('test_new_core_matmul_w_', sd.test_new_core_matmul, [deepcopy(DICT1_N), deepcopy(DICT2_N)], {'DICT2_TRANSPOSE': DICT2_N_T}),
)
















