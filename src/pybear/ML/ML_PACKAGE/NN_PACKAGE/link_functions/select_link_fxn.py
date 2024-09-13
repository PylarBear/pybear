from data_validation import validate_user_input as vui
from ML_PACKAGE.NN_PACKAGE.link_functions import link_fxns as lf
from ML_PACKAGE.NN_PACKAGE.print_results import print_nn_config as pnnc
from general_list_ops import list_select as ls

#CALLED BY nn_config
def select_link_fxn(ARRAY_OF_NODES, LINK_FXNS, SELECT_LINK_FXN, NEURONS):

    pnnc.print_nn_config(SELECT_LINK_FXN, NEURONS)


    #ARRAY NUMBER IS ASSIGNED BASED ON ORDER OF MULTIPLICATION IN NN
    #MEANING RIGHT-TO-LEFT MULT IN NN IS LEFT-TO-RIGHT THRU ARRAY_OF_NODES, NEURONS, SELECT_LINK_FXN

    while True:

        if SELECT_LINK_FXN[-1].upper() in lf.define_multi_out_links():
            print(f'LAST NODE HAS {SELECT_LINK_FXN[-1]} AND CANNOT BE CHANGED')
            node_idx = ls.list_single_select(
                [x for x in range(len(ARRAY_OF_NODES)-1)],
                'Select node to change link function',
                'idx'
            )[0]

        else:
            node_idx = ls.list_single_select(
                [x for x in range(len(ARRAY_OF_NODES))],
                'Select node to change link function',
                'idx'
            )[0]

        link_idx = ls.list_single_select(LINK_FXNS, f'Select new link # > ', 'idx')[0]

        print(f'\nChanging NODE{node_idx} from {SELECT_LINK_FXN[node_idx]} to {LINK_FXNS[link_idx]}\n')
        # CREATE DUMMY TO DISPLAY B4 CHANGE IS ACCEPTED
        SELECT_LINK_FXN2 = SELECT_LINK_FXN.copy()
        SELECT_LINK_FXN2[node_idx] = LINK_FXNS[link_idx]
        pnnc.print_nn_config(SELECT_LINK_FXN2, NEURONS)

        if vui.validate_user_str(f'Accept link selections? (y/n) > ', 'YN') == 'Y':
            SELECT_LINK_FXN[node_idx] = LINK_FXNS[link_idx]

            break

    return SELECT_LINK_FXN





