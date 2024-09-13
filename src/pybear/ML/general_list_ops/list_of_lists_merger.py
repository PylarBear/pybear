import numpy as np


def list_of_lists_merger(LIST_OF_LISTS):

    MERGED_LIST = np.hstack(LIST_OF_LISTS).reshape((1,-1))[0]

    if not isinstance(LIST_OF_LISTS, np.ndarray):
        MERGED_LIST = MERGED_LIST.tolist()

    return MERGED_LIST




if __name__=='__main__':

    LIST = [[0,1,2],[3,4,5]]
    NP = np.arange(0,6,dtype=int).reshape((2,-1))


    # TEST 1
    MERGED_LIST = list_of_lists_merger(LIST)
    if not np.array_equiv(MERGED_LIST, [0,1,2,3,4,5]):
        raise Exception(f'\033[91mTEST 1 ERROR\033[0m')
    print()

    # TEST 2
    MERGED_NP = list_of_lists_merger(NP)
    if not np.array_equiv(MERGED_NP, [0,1,2,3,4,5]):
        raise Exception(f'\033[91mTEST 2 ERROR\033[0m')
    print()


    # TEST 3
    TARGET_VECTOR = np.random.randint(0, 10, (3, 5))
    OUTPUT_VECTOR = np.random.randint(0, 10, (3, 5))
    TEST = np.corrcoef(list_of_lists_merger(TARGET_VECTOR), list_of_lists_merger(OUTPUT_VECTOR))[0][1]**2

    # TEST 4`
    TARGET_VECTOR = np.random.randint(0, 10, 10)
    OUTPUT_VECTOR = np.random.randint(0, 10, 10)
    TEST = np.corrcoef(list_of_lists_merger(TARGET_VECTOR), list_of_lists_merger(OUTPUT_VECTOR))[0][1]**2



    print(f'\n\n\033[92m*** ALL TESTS PASSED *** \033[0m')

