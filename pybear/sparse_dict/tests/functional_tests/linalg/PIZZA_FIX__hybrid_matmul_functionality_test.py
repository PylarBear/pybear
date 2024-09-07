# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

import pytest

import numpy as np
import sparse_dict as sd


pytest.skip(reason=f'24_09_07_06_57_00 needs a lot of work', allow_module_level=True)


class TestHybridMatmul:

    def test_hybrid_matmul(self):

        LIST1 = np.array([[0,1,2],[3,4,5]]).astype(np.int8)
        DICT1 = sd.zip_list_as_py_float(LIST1)
        LIST1_T = LIST1.transpose()
        DICT1_T = sd.zip_list_as_py_float(LIST1_T)
        LIST2 = np.array([[1,2],[3,4],[5,6]]).astype(np.int8)
        DICT2 = sd.zip_list_as_py_float(LIST2)
        LIST2_T = LIST2.transpose()
        DICT2_T = sd.zip_list_as_py_float(LIST2_T)

        # ANSWERS ARE CONVERTED TO np.array.astype(float64) DURING TEST
        ANSWER_KEY = {
                        'AB_AS_ROW': [[13,16],[40,52]],
                        'AB_AS_COLUMN': [[13,40],[16,52]],
                        'ATBT_AS_ROW': [[6,12,18],[9,19,29],[12,26,40]],
                        'ATBT_AS_COLUMN': [[6,9,12],[12,19,26],[18,29,40]],

                        'BTAT_AS_ROW': [[13,40],[16,52]],
                        'BTAT_AS_COLUMN': [[13,16],[40,52]],
                        'BA_AS_ROW': [[6,9,12],[12,19,26],[18,29,40]],
                        'BA_AS_COLUMN': [[6,12,18],[9,19,29],[12,26,40]]
                      }

        """ hybrid_matmul(LIST1,
                          DICT1,
                          return_as='SPARSE_DICT',
                          return_orientation='ROW') """

        MASTER_DATA = [(LIST1, DICT2, None), (LIST2, DICT1, None), (LIST1_T, DICT2_T, None), (LIST2_T, DICT1_T, None),
                        (DICT1, LIST2, None), (DICT2, LIST1, None), (DICT1_T, LIST2_T, None), (DICT2_T, LIST1_T, None),
                       (LIST1, DICT2, DICT2_T), (LIST2, DICT1, DICT1_T), (LIST1_T, DICT2_T, DICT2), (LIST2_T, DICT1_T, DICT1),
                       (DICT1, LIST2, LIST2_T), (DICT2, LIST1, LIST1_T), (DICT1_T, LIST2_T, LIST2), (DICT2_T, LIST1_T, LIST1)]

        MASTER_DATA_DESC = ["AB", "BA", "ATBT", "BTAT",
                            "AB", "BA", "ATBT", "BTAT",
                            "AB", "BA", "ATBT", "BTAT",
                            "AB", "BA", "ATBT", "BTAT"]

        total_trials = len(MASTER_DATA) * 2 * 2

        ctr = 0
        for desc, (OBJ1, OBJ2, OBJ3) in zip(MASTER_DATA_DESC, MASTER_DATA):
            for return_as in ['ARRAY', 'SPARSE_DICT']:
                for return_orientation in ['ROW', 'COLUMN']:
                    ctr += 1
                    print(f'\nTrial {ctr} of {total_trials}')
                    print(f'expected desc = {desc}')
                    print(f'expected return_as = {return_as}')
                    print(f'expected orientation = {return_orientation}')

                    print(f'PIZZA PRINT OBJ1:')
                    print(OBJ1)
                    print(f'PIZZA PRINT OBJ2:')
                    print(OBJ2)
                    print(f'PIZZA PRINT OBJ3:')
                    print(OBJ3)
                    print()

                    ACT_ANSWER = sd.hybrid_matmul(OBJ1,
                                                  OBJ2,
                                                  LIST_OR_DICT2_TRANSPOSE=OBJ3,
                                                  return_as=return_as,
                                                  return_orientation=return_orientation)

                    EXP_ANSWER = np.array(ANSWER_KEY[f'{desc}_AS_{return_orientation}'.upper()]).astype(np.float64)

                    if return_as == 'SPARSE_DICT':
                        EXP_ANSWER = sd.zip_list_as_py_float(EXP_ANSWER)

                    _exception = False
                    if return_as == 'ARRAY':
                        if not np.array_equiv(EXP_ANSWER, ACT_ANSWER): _exception = True

                    elif return_as == 'SPARSE_DICT':
                        if not sd.core_sparse_equiv(EXP_ANSWER, ACT_ANSWER): _exception = True

                    if _exception:
                        print(f'\n\033[91mEXPECTED ANSWER:\033[0m\x1B[0m')
                        print(f'\n\033[91m{EXP_ANSWER}\033[0m\x1B[0m')
                        print()
                        print(f'\n\033[91mACTUAL ANSWER:\033[0m\x1B[0m')
                        print(f'\n\033[91m{ACT_ANSWER}\033[0m\x1B[0m')
                        raise Exception(f'\n\033[91mFAIL\033[0m\x1B[0m')
                    else:
                        print(f'\033[92mALL GOOD\033[0m\x1B[0m')

        print(f'\n\033[92m*** TEST DONE. ALL PASSED. ***\033[0m\x1B[0m\n')






















