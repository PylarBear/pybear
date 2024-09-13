

def matrix_transpose(LISTTYPES_OF_LISTTYPES):

    # TRANSPOSE = []
    # for new_list_idx in range(len(LISTTYPES_OF_LISTTYPES[0])):    # NUM LISTS NOW BECOMES PRIOR NUM ELEMENTS
    #     TRANSPOSE.append([])
    #     for new_element_idx in range(len(LISTTYPES_OF_LISTTYPES)):      # NUM ELEMENTS NOW BECOMES PRIOR NUM LISTS
    #         TRANSPOSE[new_list_idx].append(LISTTYPES_OF_LISTTYPES[new_element_idx][new_list_idx])

    return list(map(list, zip(*LISTTYPES_OF_LISTTYPES)))








if __name__ == '__main__':
    import numpy as n
    for SHAPE in [[3,4], [3,3], [4,3]]:
        TEST = n.random.randint(0, 10, SHAPE)
        print(f'{SHAPE[0]} X {SHAPE[1]}:')
        [print(_) for _ in TEST]
        print()

        TRANSPOSE = matrix_transpose(TEST)
        print(f'TRANSPOSE:')
        [print(_) for _ in TRANSPOSE]
        print()






