import sparse_dict as sd
import numpy as np



# ASSUME sd.core_sparse_dict IS ACCURATE
# TEST new_sparse_transpose EQUALITY AGAINST core_sparse_dict



ctr = 0
for _dtype in ['INT', 'FLOAT']:
    for _shape in ((1,100000), (100000,1), (50,100), (100,50), (500,1000), (1000, 500)):
        ctr += 1
        print(f'Running trial {ctr} of 12...')

        if _dtype == 'INT': TEST_DICT = sd.create_random_py_int(0, 10, _shape, 90)
        elif _dtype == 'FLOAT': TEST_DICT = sd.create_random_py_float(0, 10, _shape, 90)

        EXP_DICT = sd.core_sparse_transpose(TEST_DICT)
        ACT_DICT = sd.new_sparse_transpose(TEST_DICT)

        if not sd.safe_sparse_equiv(EXP_DICT, ACT_DICT):
            print(f'\033[91m')
            print(f'\nEXP_DICT:')
            [print(f'{idx}: {EXP_DICT[_]}') for idx, _ in enumerate(EXP_DICT)]
            print(f'\nACT_DICT:')
            [print(f'{idx}: {ACT_DICT[_]}') for idx, _ in enumerate(ACT_DICT)]
            print()
            raise Exception(f'*** ACTUAL TRANSPOSE IS NOT EQUAL TO EXPECTED ***')



print(f'\033[92m*** TEST PASSES ***\033[0m')


















































