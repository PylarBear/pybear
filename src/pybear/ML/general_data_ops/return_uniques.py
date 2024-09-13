import numpy as n


#CALLED BY multi_column_to_category, target_config, and others
def return_uniques(LIST_OBJ, UNIQUES_LIST, data_type, suppress_print='N'):
    UNIQUES_LIST.clear()
    for item in LIST_OBJ:
        if item not in UNIQUES_LIST:
            UNIQUES_LIST.append(item)

    if suppress_print == 'N':
        print(f'Number of unique values: {len(UNIQUES_LIST)}')
        total_len = 0
        for idx in range(len(LIST_OBJ)):
            total_len += len(str(LIST_OBJ[idx]))
            if total_len > 120:
                print(LIST_OBJ[:idx])
                break
        else: print(LIST_OBJ)

    if data_type.upper() in ['INT','FLOAT','BIN']:
        num_list_min = n.min(n.array(UNIQUES_LIST).astype('float64'))
        num_list_max = n.max(n.array(UNIQUES_LIST).astype('float64'))
        if suppress_print == 'N':
            print(f'Min = {num_list_min}')
            print(f'Max = {num_list_max}')
    else:
        num_list_min = ''
        num_list_max = ''


    return UNIQUES_LIST, num_list_min, num_list_max






if __name__ == '__main__':

    LIST_OBJ = [1,2,3,1,2,3,1,2,3]
    UNIQUES_LIST = []
    data_type = 'int'

    UNIQUES_LIST = return_uniques(LIST_OBJ, UNIQUES_LIST, data_type, suppress_print='N')[0]
    print(UNIQUES_LIST)



