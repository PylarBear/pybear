import sys
import numpy as np
from data_validation import validate_user_input as vui
from general_data_ops import NoImprov as ni, numpy_math as nm
from general_list_ops import list_select as ls, list_of_lists_merger as llm


##########################################################################################################################
##########################################################################################################################
##########################################################################################################################

LINK_FXNS = [
    'NONE',
    'LOGISTIC',
    'RELU',
    'SOFTMAX'
]

COST_FXNS = [
    'LEAST_SQUARES',
    'MINUS_LOG_LIKELIHOOD'
]

# FUNCTION DEFINTIONS ###################################################################################################
def formatted_print(header, OBJECT, lgth):
    print('')
    obj_len = min(10, len(OBJECT))
    row_len = min(10, len(OBJECT[0]))
    print(f'\n{header.upper()}[:{obj_len}][:{row_len}] of [{len(OBJECT)}][{len(OBJECT[0])}]:')
    [print(f'{[str(round(OBJECT[_][x], lgth)).rjust(7) for x in range(row_len)]}') for _ in range(obj_len)]


def link(INTERMEDIATE, link):
    if link == 'LOGISTIC':
        return 1 / (1 + np.exp(-INTERMEDIATE, dtype=float))
    elif link == 'RELU':
        return INTERMEDIATE * (INTERMEDIATE > 0)
    elif link == 'SOFTMAX':
        return np.transpose([np.exp(_)/np.sum(np.exp(_)) for _ in np.transpose(INTERMEDIATE)])
    elif link == 'NONE':
        return INTERMEDIATE


class NS(nm.Numpy_Math):
    pass


def erf(ARRAY_OF_NODES, OUTPUT, TARGET, cost_fxn, rglztn_factor):

    weighted_array_of_nodes_2_norm = \
                0.5 * rglztn_factor * np.sum([np.sum(np.power(X, 2)) for X in ARRAY_OF_NODES])

    if cost_fxn == 'MINUS_LOG_LIKELIHOOD' and len(TARGET) == 1:  # FOR REGULAR TARGET
        return weighted_array_of_nodes_2_norm - NS().sumf(   #MINUS TO MAKE LL POSITIVE
                        NS().addf(
                                NS().multiplyf(TARGET, np.log(OUTPUT)),
                                NS().multiplyf(NS().subtractf(1, TARGET), np.log(NS().subtractf(1, OUTPUT)))
                        )
        )

    elif cost_fxn == 'MINUS_LOG_LIKELIHOOD' and len(TARGET) > 1:  # FOR SOFTMAX TARGET

        CONSOLIDATED_OUTPUT = llm.list_of_lists_merger(
                                    [[OUTPUT[category_idx][example_idx] for \
                                    example_idx in range(len(OUTPUT[0])) if \
                                    TARGET[category_idx][example_idx] == 1] for \
                                    category_idx in range(len(OUTPUT))]
                                )

        return weighted_array_of_nodes_2_norm + NS().sumf(-np.log(CONSOLIDATED_OUTPUT)) # - TO MAKE LL POSITIVE

    elif cost_fxn == 'LEAST_SQUARES':
        return weighted_array_of_nodes_2_norm + NS().sumf(np.power(NS().subtractf(TARGET, OUTPUT), 2))


def calculate_output_and_error(DATA, ARRAY_OF_NODES, TARGET, LINKS, activation_constant, cost_fxn, rglztn_factor):
    ARRAY_OF_z_x = []
    ARRAY_OF_a_x_b4 = []
    ARRAY_OF_a_x_after = []
    INTERMEDIATE = np.transpose(DATA)
    for node_idx in range(len(ARRAY_OF_NODES)):
        # APPEND ZEROS TO a_x_b4 TO CREATE a_x_after, RIGHT BEFORE MATMUL INTO NEXT NODE
        if activation_constant == 1 and node_idx not in [0]:  #DONT APPEND 1s TO ORIGINAL DATA & OUTPUT
            INTERMEDIATE = np.insert(INTERMEDIATE, len(INTERMEDIATE), 1, axis=0)
        # APPEND THIS AUGMENTED (OR NOT AUGMENTED) MATRIX TO ARRAY_OF_a_x_after
        ARRAY_OF_a_x_after.append(INTERMEDIATE)
        # MATMUL a_x_after WITH THE FOLLOWING NODE
        INTERMEDIATE = np.matmul(ARRAY_OF_NODES[node_idx], INTERMEDIATE, dtype=float)
        # APPEND RAW MATMUL OUTPUT TO ARRAY_OF_zs
        ARRAY_OF_z_x.append(INTERMEDIATE)
        # PASS z THROUGH LINK, TURNING IT INTO a_x
        INTERMEDIATE = link(INTERMEDIATE, LINKS[node_idx])
        # APPEND OUTPUT FROM LINK TO ARRAY_OF_a_x_b4
        ARRAY_OF_a_x_b4.append(INTERMEDIATE)

        # B4 MEANS B4 APPENDING 1s, AFTER MEANS AFTER APPENDING 1s (OR NOT APPENDED IF SO CHOSEN)
        '''ARRAYS SHOULD LOOK LIKE THIS:  (USING 4 NODES JUST AS AN EXAMPLE)
        ARRAY_OF_a_x_after = [] ---> [ DATA           , a0after            , a1after           , a2after                    ]
        ARRAY OF NODES --->          [     NODE0      ,        NODE1       ,        NODE2      ,        NODE3               ]
        ARRAY_OF_z_x = [] --->       [          z0    ,             z1     ,             z2    ,             z3             ]
        ARRAY_OF_a_x_b4 = [] --->    [            a0b4,               a1b4 ,               a2b4,                a3b4(OUTPUT)]
        '''

    OUTPUT = INTERMEDIATE

    wip_error = erf(ARRAY_OF_NODES, OUTPUT, TARGET, cost_fxn, rglztn_factor)

    return OUTPUT, ARRAY_OF_z_x, ARRAY_OF_a_x_b4, ARRAY_OF_a_x_after, wip_error


def d_costfxn_wrt_output(TARGET, OUTPUT, cost_fxn):

    if cost_fxn == 'LEAST_SQUARES':
        return NS().subtractf(OUTPUT, TARGET)

    elif cost_fxn == 'MINUS_LOG_LIKELIHOOD' and len(TARGET) == 1:   #FOR REGULAR TARGET
        return NS().dividef(
                                NS().subtractf(OUTPUT, TARGET),
                                NS().multiplyf(OUTPUT, NS().subtractf(1, OUTPUT))
        )

    elif cost_fxn == 'MINUS_LOG_LIKELIHOOD' and len(TARGET) > 1:   #FOR SOFTMAX TARGET

        GRAD_ARRAY = []
        for category_idx in range(len(TARGET)):
            GRAD_ARRAY.append([])
            for example_idx in range(len(TARGET[category_idx])):
                if TARGET[category_idx][example_idx] == 1:
                    GRAD_ARRAY[-1].append(-1 / OUTPUT[category_idx][example_idx])
                elif TARGET[category_idx][example_idx] == 0:
                    GRAD_ARRAY[-1].append(1 / (1 - OUTPUT[category_idx][example_idx]))
                else:
                    raise ValueError(f'THERE IS A NON-BINARY IN TARGET VECTOR.')

        return GRAD_ARRAY


def da_dz(INTERMEDIATE, link):
    if link == 'LOGISTIC':
        return NS().multiplyf(INTERMEDIATE, NS().subtractf(1, INTERMEDIATE))

    elif link == 'SOFTMAX':
        # DERIVIATES ARE THE SAME NO MATTER WHETHER THE OUTPUT SHOULD BE A ZERO OR A ONE
        return NS().multiplyf(INTERMEDIATE, NS().subtractf(1, INTERMEDIATE))

    elif link == 'RELU':
        return (INTERMEDIATE >= 0)**2   # SQUARING TO TURN BOOLEANS INTO 0,1

    elif link == 'NONE':
        return INTERMEDIATE / INTERMEDIATE    # ALL ONES

# END FUNCTION DEFS ######################################################################################################

##########################################################################################################################
##########################################################################################################################
##########################################################################################################################

DATA =  [
    [ 1, 2, 0,-2, 0],
    [ 2,-1, 1,-2, 1],
    [ 3, 1, 0,-1, 3],
    [ 2, 0, 1,-1,-2],
    [-1, 2,-2, 3,-1],
    [ 1, 1, 3,-3, 0],
    [ 2, 0,-2,-1, 3],
    [ 6, 5, 2, 8, 1],
    [ 1, 3,-4, 4,-8],
    [ 2, 2,-2, 1,-1],
    [-8,-7,-7, 4, 4],
    [-4, 6, 9,-5, 3],
    [ 4,-2, 0, 6,-3],
]

_ = vui.validate_user_str(f'Use regular(r) logistic(l) or softmax(s) target? > ', 'LRS')
if _ == 'L':
    TARGET = [[ 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0]]
elif _ == 'R':
    TARGET = [[-8, 4, 7,-3, 0,12, 2,-2, 9, 1,-4, 6, 3]]
elif _ == 'S':
    TARGET = [
                [ 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0],
                [ 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0],
                [ 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]
    ]


while True:

    cost_fxn = ls.list_single_select(COST_FXNS, 'Select cost function', 'value')[0]

    while True:

        NEURONS = []
        node_cnt = vui.validate_user_int(f'\nEnter number of nodes (integer > 0) > ', min=1, max=10)

        for node_idx in range(node_cnt - 1):  # LAST NODE IS DETERMINED BY FINAL LINK & TARGET VECTOR
            NEURONS.append(vui.validate_user_int(f'Enter number of neurons for NODE {node_idx} > ', min=1, max=1e3))

        if len(TARGET) == 1:
            NEURONS.append(1)  #NUMBER OF NEURONS FOR LAST NODE (NOT CONFIGURED FOR SOFTMAX YET)
            print(f'# Neurons for NODE2 must be 1.')
        elif len(TARGET) > 1:
            NEURONS.append(len(TARGET))
            print(f'Final link must be SOFTMAX and Neurons for NODE{node_cnt-1} must be {len(TARGET)}.')

        print(f'\nNEURON selections look like:')
        [print(f'NODE{idx})'.ljust(8,' ') + f'{NEURONS[idx]}')  for idx in range(len(NEURONS))]

        if vui.validate_user_str(f'Accept NODE / NEURON config? (y/n) > ', 'YN') == 'Y':
            break

    print(f'\nNEURONS = {NEURONS}\n')

    while True:

        [print(f'{idx + 1}) {LINK_FXNS[idx]}') for idx in range(len(LINK_FXNS))]
        LINKS = [LINK_FXNS[vui.validate_user_int(f'Select link for NODE{idx} > ', min=1,max=len(LINK_FXNS))-1].upper() \
                        for idx in range(node_cnt-1)]

        if len(TARGET) == 1:
            NEW_LINK_FXNS = [LINK_FXNS[idx] for idx in range(len(LINK_FXNS)) if LINK_FXNS[idx] not in 'SOFTMAX']
            [print(f'{idx + 1}) {NEW_LINK_FXNS[idx]}') for idx in range(len(NEW_LINK_FXNS))]
            # LINKS.append([NEW_LINK_FXNS[vui.validate_user_int(f'Select link for NODE{idx} > ', min=1,max=len(LINK_FXNS))-1].upper() \
            #                 for idx in range(node_cnt)])
            LINKS.append(NEW_LINK_FXNS[vui.validate_user_int(f'Select link for NODE{node_cnt-1} > ', min=1,max=len(NEW_LINK_FXNS))-1].upper())

        elif len(TARGET) > 1:
            print(f'\nFINAL LINK MUST BE SOFTMAX WITH {len(TARGET)} CATEGORIES.\n')
            LINKS.append('SOFTMAX')

        print(f'Link functions look like:')
        [print(f'NODE{idx}) {LINKS[idx]}') for idx in range(len(LINKS))]
        if vui.validate_user_str(f'Accept link selections? (y/n) > ', 'YN') == 'Y':
            break

    if vui.validate_user_str(f'Use intercept? (Append a column of 1s to DATA) (y/n) > ', 'YN') == 'Y':
        [DATA[ex_idx].append(1) for ex_idx in range(len(DATA))]

    if node_cnt > 1:
        if vui.validate_user_str(f'Use activation constant on all activations? (Append a row of 1s to activation matrices) (y/n) > ', 'YN') == 'Y':
            activation_constant = 1
        else:
            activation_constant = 0
    else:
        activation_constant = 0

    COLUMNS = [len(DATA[0])]
    [COLUMNS.append(activation_constant + NEURONS[node_idx - 1]) for node_idx in range(1, node_cnt)]

    print(f'COLUMNS = {COLUMNS}')
    while True:
        learning_rate = vui.validate_user_float(f'Enter learning rate > ', min=-1, max=1)
        # old_learning_rate = learning_rate  # THIS IS A RELIC OF STEP-WISE INCREASING LEARNING RATE LOOP
        m_weight = vui.validate_user_float(f'Enter momentum weight (fraction assign to old gradient) > ', min=0, max=1)

        seed_method = vui.validate_user_str(f'Seed NN with constant(c) or ndist(n)? > ', 'CN')
        if seed_method == 'C':
            seed = vui.validate_user_float(f'Enter seed constant > ', min=0, max=1e3)

        if seed_method == 'N':
            mu = vui.validate_user_float(f'Enter mean for node seed ndist > ')
            stdev = vui.validate_user_float(f'Enter stdev for node seed ndist > ', min=1e-10)

        rglztn_factor = vui.validate_user_float(f'Enter regularization constant (0 for no regularization) > ', min=0,
                                                max=1e10)

        if vui.validate_user_str(f'Accept hyperparameters? (y/n) > ', 'YN') == 'Y':
            break

    pct_change = vui.validate_user_float(f'Enter min % change required to avert kill > ', min=0, max=1000)
    conv_kill = vui.validate_user_int(f'Enter iterations for convergence kill > ', min=1)
    no_improv_ctr, best_value = 0, float('inf')

    while True:
        ARRAY_OF_NODES = []
        for node_idx in range(node_cnt):
            if seed_method == 'C':
                ARRAY_OF_NODES.append(
                    [[seed for col in range(COLUMNS[node_idx])] for _ \
                            in range(NEURONS[node_idx])]
                )
            elif seed_method == 'N':
                ARRAY_OF_NODES.append(
                    [[np.random.normal(mu, stdev) for col in range(COLUMNS[node_idx])] for _ \
                            in range(NEURONS[node_idx])])

        print(f'\nNODES LOOK LIKE:\n')
        for node_idx in range(node_cnt):
            formatted_print(f'NODE{node_idx}', ARRAY_OF_NODES[node_idx], 4)
        print('')


        while True:
            ARRAY_OF_GRADIENTS = np.array([0 for _ in range(len(ARRAY_OF_NODES))], dtype=object)  # SEED ARRAY_OF_GRADIENTS TO ESTABLISH INDICES
            for itr in range(1, 1 + vui.validate_user_int(f'Enter number of iterations > ', min=1, max=1e5)):
                print(f'\nPERFORMING ITERATION {itr}')
                INTERMEDIATE = np.transpose(DATA)

                # FORWARD PROP #####################################################################################################
                OUTPUT, ARRAY_OF_z_x, ARRAY_OF_a_x_b4, ARRAY_OF_a_x_after, total_error = calculate_output_and_error(
                    DATA, ARRAY_OF_NODES, TARGET, LINKS, activation_constant, cost_fxn, rglztn_factor)

                print(f'ITERATION {itr} STARTING ERROR: {total_error}')
                # END FORWARD PROP #################################################################################################

                # BACK PROP ###############################################################################################
                # BUILD cost_fxn DERIVATIVE WRT OUTPUT
                dL_dan = d_costfxn_wrt_output(TARGET, OUTPUT, cost_fxn)

                # BUILD OUTPUT DERIVATIVE WRT z_n
                dan_dzn = da_dz(OUTPUT, LINKS[-1])

                # DOT dl_dan WITH dan_dzn
                DOTTED_FINAL_DERIVATIVES = NS().multiplyf(dL_dan, dan_dzn)

                # SET THIS HERE FOR no_improv
                old_total_error = total_error

                for target_node_idx in range(len(ARRAY_OF_NODES)-1,-1,-1):
                    # CALCULATE THE ODDBALL CASE OF THE LAST GRADIENT ONCE, THEN CARRY THOUGH THE LOOP FOR REMAINING GRADS
                    if target_node_idx == len(ARRAY_OF_NODES)-1:
                        ARRAY_OF_GRADIENTS[target_node_idx] = np.matmul(DOTTED_FINAL_DERIVATIVES,
                                                          np.transpose(ARRAY_OF_a_x_after[-1]),
                                                          dtype=float)

                        STUB = DOTTED_FINAL_DERIVATIVES

                    else:
                        if activation_constant == 1:
                            INTERMEDIATE2 = np.array([X[:-1] for X in ARRAY_OF_NODES[target_node_idx+1]])
                        else:
                            INTERMEDIATE2 = ARRAY_OF_NODES[target_node_idx+1]

                        INTERMEDIATE2 = np.matmul(np.transpose(INTERMEDIATE2), STUB)
                        INTERMEDIATE2 = NS().multiplyf(da_dz(ARRAY_OF_a_x_b4[target_node_idx], LINKS[target_node_idx]), INTERMEDIATE2)
                        STUB = INTERMEDIATE2
                        ARRAY_OF_GRADIENTS[target_node_idx] = np.matmul(INTERMEDIATE2, np.transpose(ARRAY_OF_a_x_after[target_node_idx]),
                                                                       dtype=float)
                # END BACK PROP ############################################################################################

                # MOMENTUM ################################################################################################

                if itr == 1:
                    OLD_ARRAY_OF_GRADIENTS = ARRAY_OF_GRADIENTS.copy()
                #else:  (#THIS IS NOT USABLE CODE, THIS IS HERE JUST TO SHOW THE LOGIC)
                    #OLD_ARRAY_OF_GRADIENTS IS TAKEN FROM THE PREVIOUS ITERATION, AS SET BELOW

                # ADJUST WITH MOMENTUM, TECHNICALLY ONLY AFTER FIRST itr
                # HAVE TO DO IT THE LONG WAY AGAIN BECAUSE OF RAGGED ARRAY / NUMPY MATH BLOWUPS
                for node_idx in range(len(ARRAY_OF_GRADIENTS)):
                    for neuron_idx in range(len(ARRAY_OF_GRADIENTS[node_idx])):
                        ARRAY_OF_GRADIENTS[node_idx][neuron_idx] = NS().addf(
                            m_weight * OLD_ARRAY_OF_GRADIENTS[node_idx][neuron_idx], \
                            (1 - m_weight) * ARRAY_OF_GRADIENTS[node_idx][neuron_idx])

                # SET OLD GRADIENT FOR NEXT LOOP
                OLD_ARRAY_OF_GRADIENTS = ARRAY_OF_GRADIENTS

                # END MOMENTUM ##############################################################################################

                # UPDATE NN PARAMETERS #######################################################################################
                for node_idx in range(len(ARRAY_OF_NODES)):   #HASHED OUT FORMULA BELOW IS TO USE A SMALLER LEARNING RATE GETTING FARTHER INTO ARRAY OF NODES
                    ARRAY_OF_NODES[node_idx] = NS().subtractf(NS().multiplyf(1 - learning_rate * rglztn_factor, ARRAY_OF_NODES[node_idx]),
                             NS().multiplyf(learning_rate, ARRAY_OF_GRADIENTS[node_idx])) #/10**(len(ARRAY_OF_NODES)-node_idx-1), ARRAY_OF_GRADIENTS[node_idx]))
                # END UPDATE NN PARAMETERS ####################################################################################

                    OUTPUT, ARRAY_OF_z_x, ARRAY_OF_a_x_b4, ARRAY_OF_a_x_after, total_error = calculate_output_and_error(
                            DATA, ARRAY_OF_NODES, TARGET, LINKS, activation_constant, cost_fxn, rglztn_factor)

                    print(f'ITERATION {itr}) TOTAL ERROR AFTER UPDATE TO NODE {node_idx} = {total_error}')

                OUTPUT, ARRAY_OF_z_x, ARRAY_OF_a_x_b4, ARRAY_OF_a_x_after, total_error = calculate_output_and_error(
                    DATA, ARRAY_OF_NODES, TARGET, LINKS, activation_constant, cost_fxn, rglztn_factor)

                ####################################################################################################################

                print(f'ITR {itr} TOTAL ERROR AFTER ALL NODE UPDATES: {total_error}')

                no_improv_ctr, best_value = ni.no_improv('MIN', total_error, itr, 0, no_improv_ctr, best_value, pct_change)

                if no_improv_ctr == conv_kill:
                    if vui.validate_user_str(
                            f'GD as not achieved greater than {pct_change}% change for {conv_kill} consecutive iterations.  Break? (y/n) > ',
                            'YN') == 'Y':
                        break
                    else:
                        no_improv_ctr = 0

                # DIVERGENCE BREAK
                if total_error > old_total_error:
                    print(f'\nError has diverged on iteration {itr}.')
                    div_action = vui.validate_user_str(
                        f'Enter a new learning rate(r) or abandon this trial(a)? > ', 'AR')
                    if div_action == 'R':
                        learning_rate = vui.validate_user_float(
                            f'Enter new learning rate (old rate was {learning_rate}) > ', min=1e-15)
                        # old_learning_rate = learning_rate
                    elif div_action == 'A':
                        break

            print(f'\nUser specified iterations are complete or busted.\n')
            while True:
                user_prompt = vui.validate_user_str(
                    f'Run again with ... same settings with node reset(r) without node reset(w) different settings(d) ... ' + \
                    'print OUTPUT & TARGET(p) ...or quit(q)? > ', 'DQRPW')
                if user_prompt == 'P':
                    print('OUTPUT:')
                    [print([str(round(x, 3)).rjust(5, " ") for x in category]) for category in OUTPUT]
                    print('TARGET:')
                    [print([str(round(x, 3)).rjust(5, " ") for x in category]) for category in TARGET]
                    print(f'RSQ: ' +
                          f'{np.corrcoef(llm.list_of_lists_merger(TARGET), llm.list_of_lists_merger(OUTPUT))[0, 1] ** 2}')
                else:
                    break

            if user_prompt == 'Q':
                sys.exit(f'User terminated.')
            elif user_prompt == 'D':
                break
            elif user_prompt == 'R':
                break
            elif user_prompt == 'W':
                if vui.validate_user_str(f'Change learning rate? (y/n) > ', 'YN') == 'Y':
                    learning_rate = vui.validate_user_float(f'Enter learning rate (last was {learning_rate}) > ', min=1e-15)
                    # old_learning_rate = learning_rate
                if vui.validate_user_str(f'Change regularization constant? (y/n) > ', 'YN') == 'Y':
                    rglztn_factor = vui.validate_user_float(f'Enter regularization constant (last was {rglztn_factor}) > ', min=1e-15)
                continue

        if user_prompt == 'R':
            continue

        if user_prompt == 'D':
            break









# # FORWARD PROP
# for node_idx in range(len(ARRAY_OF_NODES)):
#
#     ################################################################################################################
#     # CAPTURE a0 RIGHT BEFORE IT GOES INTO THE SECOND NODE (AND RIGHT BEFORE THE 1 GETS APPENDED)###################
#     # (a0 IS THE ACTIVATION COMING OUT OF NODE0 (THE 1ST NODE)) ####################################################
#     if node_idx == 1:
#         a0_b4 = INTERMEDIATE
#     # END a0_b4 CAPTURE ###############################################################################################
#     ################################################################################################################
#
#     ################################################################################################################
#     # CAPTURE a1 RIGHT BEFORE IT GOES INTO THE THIRD NODE (AND RIGHT BEFORE THE 1 GETS APPENDED)#####################
#     # (a1 IS THE ACTIVATION COMING OUT OF NODE1 (THE 2ND NODE)) ####################################################
#     if node_idx == 2:
#         a1_b4 = INTERMEDIATE
#     # END a1_b4 CAPTURE ###############################################################################################
#     ################################################################################################################
#
#     # COLLECT a0 & a1 BEFORE APPENDING THE 1s
#     if activation_constant == 1 and node_idx not in [0]:
#         INTERMEDIATE = np.insert(INTERMEDIATE, len(INTERMEDIATE), 1, axis=0)
#
#     ################################################################################################################
#     # CAPTURE a0 RIGHT BEFORE IT GOES INTO THE SECOND NODE (AND RIGHT AFTER THE 1 GETS APPENDED)###################
#     # (a0 IS THE ACTIVATION COMING OUT OF NODE0 (THE 1ST NODE)) ####################################################
#     if node_idx == 1:
#         a0_after = INTERMEDIATE
#     # END a0_after CAPTURE #########################################################################################
#     ################################################################################################################
#
#     ################################################################################################################
#     # CAPTURE a1 RIGHT BEFORE IT GOES INTO THE THIRD NODE (AND RIGHT AFTER THE 1 GETS APPENDED)#####################
#     # (a1 IS THE ACTIVATION COMING OUT OF NODE1 (THE 2ND NODE)) ####################################################
#     if node_idx == 2:
#         a1_after = INTERMEDIATE
#     # END a1_after CAPTURE ###############################################################################################
#     ################################################################################################################
#
#     INTERMEDIATE = np.matmul(ARRAY_OF_NODES[node_idx], INTERMEDIATE, dtype=float)
#
#     # ################################################################################################################
#     # # CAPTURE z1 RIGHT BEFORE IT GOES INTO LINK FXN ################################################################
#     # # (z1 IS THE OUTPUT OF NODE1 PRE ACTIVATION (THE 2ND NODE)) ####################################################
#     # if node_idx == 1:
#     #     z1 = INTERMEDIATE
#     # # END z1 CAPTURE ###############################################################################################
#     # ################################################################################################################
#
#     INTERMEDIATE = link(INTERMEDIATE, 'LOGISTIC')
#
# OUTPUT = INTERMEDIATE
#
# total_error = erf(ARRAY_OF_NODES, OUTPUT, TARGET, 'minus_log_likelihood'.upper(), rglztn_factor)
# print(f'\nITR {itr} TOTAL ERROR BEFORE UPDATE: {total_error}')


'''

                #FOR MOMENTUM
                if itr > 1:
                    OLD_NODE2_UPDATE_VECTOR = NODE2_UPDATE_VECTOR

                # BUILD AN UPDATE VECTOR SEED
                NODE2_UPDATE_VECTOR = np.array([0 for _ in range(len(a1_after[0]))], dtype=float)

                # ADD ALL WEIGHTED ROWS IN a1
                for row_idx in range(len(a1_after)):
                    NODE2_UPDATE_VECTOR = NS().addf(a1_after[row_idx], NODE2_UPDATE_VECTOR)

                # ADJUST WITH MOMENTUM, ONLY AFTER FIRST itr
                if itr > 1:
                    NODE2_UPDATE_VECTOR = m_weight * OLD_NODE2_UPDATE_VECTOR + (1 - m_weight) * NODE2_UPDATE_VECTOR


                ########################################################################################################
                # LOOP FOR INCREASING LEARNING RATE MULTIPLE TO FIND TRUE MIN ##########################################
                ORIGINAL_NODE2 = ARRAY_OF_NODES[-1].copy()
                ERROR_HOLDER = []
                NODE2_HOLDER = []

                lr_trial_limit = 1000
                for _ in range(lr_trial_limit):
                    # print(f'LEARNING RATE = {learning_rate}, total_error going in = {total_error}')
                    # NODE2_UPDATE_VECTOR = NS().multiplyf(learning_rate, NODE2_UPDATE_VECTOR)
                    DUMMY_NODE2_UPDATE_VECTOR = NS().multiplyf(learning_rate, NODE2_UPDATE_VECTOR)

                    # >>>>>>>>> USING SUBTRACT HERE <<<<<<<<<<<<<<
                    # ARRAY_OF_NODES[-1] = NS().subtractf(ARRAY_OF_NODES[-1], NODE2_UPDATE_VECTOR)

                    ARRAY_OF_NODES[-1] = NS().subtractf(ARRAY_OF_NODES[-1], DUMMY_NODE2_UPDATE_VECTOR)

                # JUST TO CALCULATE ERROR AFTER LAST NODE UPDATE ##################################################################

                    OUTPUT, ARRAY_OF_z_x, ARRAY_OF_a_x_b4, ARRAY_OF_a_x_after, total_error = calculate_output_and_error(
                        DATA, ARRAY_OF_NODES, TARGET, LINKS, activation_constant, cost_fxn, rglztn_factor)

                    ####################################################################################################################

                    ERROR_HOLDER.append(total_error)
                    NODE2_HOLDER.append(ARRAY_OF_NODES[-1])

                    if len(ERROR_HOLDER) > 1:
                        if ERROR_HOLDER[-1] > ERROR_HOLDER[-2]:
                            total_error = ERROR_HOLDER[-2]
                            ARRAY_OF_NODES[-1] = NODE2_HOLDER[-2]
                            ORIGINAL_NODE2 = ARRAY_OF_NODES[-1]
                            ERROR_HOLDER.clear()
                            NODE2_HOLDER.clear()
                            learning_rate = old_learning_rate
                            break
                        elif _ == lr_trial_limit - 1:
                            total_error = ERROR_HOLDER[-1]
                            ARRAY_OF_NODES[-1] = NODE2_HOLDER[-1]
                            ORIGINAL_NODE2 = ARRAY_OF_NODES[-1]
                            ERROR_HOLDER.clear()
                            NODE2_HOLDER.clear()
                            learning_rate = old_learning_rate
                            break
                        else:
                            learning_rate += old_learning_rate
                            # ARRAY_OF_NODES[-1] = ORIGINAL_NODE2
                    else:
                        learning_rate += old_learning_rate
                # END LOOP FOR INCREASING learning_rate #################################################################
                ####################################################################################################################

                # RESET learning_rate BACK TO OLD AFTER INCREMENTING IT FOR LAST NODE
                learning_rate = old_learning_rate



                # END FIRST SHOT AT BACKPROP ON LAST NODE ##########################################################################
                ####################################################################################################################
                ####################################################################################################################


#                 ####################################################################################################################
#                 ####################################################################################################################
#                 # FIRST SHOT AT BACKPROP ON PENULTIMATE NODE (LOOKS LIKE THIS WORKS) ###############################################
#
#                 # FOR MOMENTUM
#                 if itr > 1:
#                     OLD_NODE1_UPDATE_MATRIX = NODE1_UPDATE_MATRIX
#
#                 if activation_constant == 1:
#                     FIRST_TERM = np.array(ARRAY_OF_NODES[-1][0][:-1])  #GET EVERYTHING FROM LAST NODE EXCEPT THE CONSTANT
#                 else:
#                     FIRST_TERM = np.array(ARRAY_OF_NODES[-1][0])
#                 SECOND_TERM = NS().multiplyf(a1_b4, NS().subtractf(1, a1_b4))
#                 THIRD_TERM = NS().subtractf(OUTPUT, TARGET)
#                 FOURTH_TERM = np.transpose(a0_after)
#
#                 #PROPOGATE first_term ELEMENTWISE INTO second_term, WITH MORE GYRATIONS TO ACCOMPLISH THIS
#                 SECOND_TERM = np.transpose(SECOND_TERM)
#                 for column_idx in range(len(SECOND_TERM)):
#                     SECOND_TERM[column_idx] = NS().multiplyf(SECOND_TERM[column_idx], FIRST_TERM)
#                 SECOND_TERM = np.transpose(SECOND_TERM)
#                 #PROPOGATE third_term ELEMENTWISE INTO second_term
#                 for row_idx in range(len(SECOND_TERM)):
#                     SECOND_TERM[row_idx] = NS().multiplyf(SECOND_TERM[row_idx], THIRD_TERM)
#                 #MATMUL second_term & fourth_term
#                 NODE1_UPDATE_MATRIX = np.matmul(SECOND_TERM, FOURTH_TERM, dtype=float)
#
#
#                 # ADJUST WITH MOMENTUM
#                 if itr > 1:
#                     NODE1_UPDATE_MATRIX = NS().addf(
#                                                 NS().multiplyf(m_weight, OLD_NODE1_UPDATE_MATRIX),
#                                                 NS().multiplyf(1 - m_weight, NODE1_UPDATE_MATRIX)
#                     )
#
# #>>>>>>>>> REMEMBER THERE IS ANOTHER FACTOR APPLIED TO THE LEARNING RATE HERE FOR NODE[-2] <<<<<<<<<<<<<
#                 # >>>>>>>>> USING SUBTRACT HERE <<<<<<<<<<<<<<
#                 ARRAY_OF_NODES[1] = NS().subtractf(ARRAY_OF_NODES[-2], NS().multiplyf(.01*learning_rate, NODE1_UPDATE_MATRIX))
#
#
#                 ###### CALCULATE NEW ERROR, UPDATE a0s AND a1s #################################################
#                 INTERMEDIATE = np.transpose(DATA)
#                 for node_idx in range(len(ARRAY_OF_NODES)):
#
#                     ################################################################################################################
#                     # CAPTURE a0 RIGHT BEFORE IT GOES INTO THE SECOND NODE (AND RIGHT BEFORE THE 1 GETS APPENDED)###################
#                     # (a0 IS THE ACTIVATION COMING OUT OF NODE0 (THE 1ST NODE)) ####################################################
#                     if node_idx == 1:
#                         a0_b4 = INTERMEDIATE
#                     # END a0_b4 CAPTURE ###############################################################################################
#                     ################################################################################################################
#
#                     ################################################################################################################
#                     # CAPTURE a1 RIGHT BEFORE IT GOES INTO THE THIRD NODE (AND RIGHT BEFORE THE 1 GETS APPENDED)#####################
#                     # (a1 IS THE ACTIVATION COMING OUT OF NODE1 (THE 2ND NODE)) ####################################################
#                     if node_idx == 2:
#                         a1_b4 = INTERMEDIATE
#                     # END a1_b4 CAPTURE ###############################################################################################
#                     ################################################################################################################
#
#                     if activation_constant == 1 and node_idx not in [0]:
#                         INTERMEDIATE = np.insert(INTERMEDIATE, len(INTERMEDIATE), 1, axis=0)
#
#                     ################################################################################################################
#                     # CAPTURE a0 RIGHT BEFORE IT GOES INTO THE SECOND NODE (AND RIGHT AFTER THE 1 GETS APPENDED)###################
#                     # (a0 IS THE ACTIVATION COMING OUT OF NODE0 (THE 1ST NODE)) ####################################################
#                     if node_idx == 1:
#                         a0_after = INTERMEDIATE
#                     # END a0_after CAPTURE #########################################################################################
#                     ################################################################################################################
#
#                     ################################################################################################################
#                     # CAPTURE a1 RIGHT BEFORE IT GOES INTO THE THIRD NODE (AND RIGHT AFTER THE 1 GETS APPENDED)#####################
#                     # (a1 IS THE ACTIVATION COMING OUT OF NODE1 (THE 2ND NODE)) ####################################################
#                     if node_idx == 2:
#                         a1_after = INTERMEDIATE
#                     # END a1_after CAPTURE ###############################################################################################
#                     ################################################################################################################
#
#                     INTERMEDIATE = np.matmul(ARRAY_OF_NODES[node_idx], INTERMEDIATE, dtype=float)
#
#                     # ################################################################################################################
#                     # # CAPTURE z1 RIGHT BEFORE IT GOES INTO LINK FXN ################################################################
#                     # # (z1 IS THE OUTPUT OF NODE1 PRE ACTIVATION (THE 2ND NODE)) ####################################################
#                     # if node_idx == 1:
#                     #     z1 = INTERMEDIATE
#                     # # END z1 CAPTURE ###############################################################################################
#                     # ################################################################################################################
#
#                     INTERMEDIATE = link(INTERMEDIATE, 'LOGISTIC')
#
#                 OUTPUT = INTERMEDIATE
#
#                 total_error = erf(ARRAY_OF_NODES, OUTPUT, TARGET, 'minus_log_likelihood'.upper(), rglztn_factor)
#
#                 print(f'ITR {itr} TOTAL ERROR AFTER UPDATE TO PENULTIMATE NODE: {total_error}')
#
#                 # END FIRST SHOT AT BACKPROP ON PENULTIMATE NODE ###################################################################
#                 ####################################################################################################################
#                 ####################################################################################################################
#

                


'''
