
#COMES FROM FILTERING PACKAGE:
#METADATA_KEEP, START_DATE, END_DATE, FILTER_MATRIX

#COMES FROM MATRIX SETUP PACKAGE:
#interaction, int_cutoff, intercept, BIG_MATRIX

#COMES FROM NN_CONFIG PACKAGE:
#nodes, NEURONS, non_neg_coeffs, node_seed, SELECT_LINK_FXN, gd_method, lr_method, conv_method, LEARNING_RATE, gd_iterations


#CALLED BY NN
def print_settings(METADATA_KEEP, FILTER_MATRIX, START_DATE, END_DATE, BIG_MATRIX, interaction, int_cutoff, intercept,
                    nodes, NEURONS, node_seed, SELECT_LINK_FXN, non_neg_coeffs, gd_method, lr_method, conv_method,
                   gd_iterations):

    print(f'''
SETUP:
    EXTRACT:
        ATTRIBUTE  KEEP  CUTOFF  "OTHERS"''')
    for line in METADATA_KEEP:
        print(f'          {line}')

    print(f'''
    FILTERING:
        BY DATE:   {START_DATE} - {END_DATE}
        BY CATEGORY:
            ATTRIBUTE  CATEGORY  INCL/EXCL  KEEP/DEL''')
    for row in FILTER_MATRIX:
        print(f'                  {row}')

    print(f'''

    Use Interactions:   {interaction}
    Interaction cutoff: {int_cutoff}
    Use Intercept:      {intercept}
    
BIG_MATRIX info:''')
    KEPT_DICT = {}
    KEPT_IDENTIFIER_DICT = {}
    total_hits = 0
    for column in range(2, len(WIP_MATRIX)):
        hits = sum(WIP_MATRIX[column][2:])
        total_hits += hits

        if str(f'{WIP_MATRIX[column][0]}') not in KEPT_IDENTIFIER_DICT:
            KEPT_IDENTIFIER_DICT[str(f'{WIP_MATRIX[column][0]}')] = 0
        KEPT_IDENTIFIER_DICT[str(f'{WIP_MATRIX[column][0]}')] += hits

        if str(f'{WIP_MATRIX[column][0]}') not in KEPT_DICT:
            KEPT_DICT[str(f'{WIP_MATRIX[column][0]}')] = []
        if str(f'{WIP_MATRIX[column][1]}') not in KEPT_DICT[str(f'{WIP_MATRIX[column][0]}')]:
            KEPT_DICT[str(f'{WIP_MATRIX[column][0]}')].append(str(f'{WIP_MATRIX[column][1]}'))

    print(f'')
    print(f'Hits still in BIG_MATRIX:')
    print(f'{KEPT_IDENTIFIER_DICT}')
    print(f'')
    print(f'List of categories still in BIG_MATRIX')
    for item in KEPT_DICT:
        print(f'{item}: {KEPT_DICT[item]}')

    print(f'''    
NN INFO:
    Nodes:              {nodes}
    Node Seed:          {node_seed}
    Neurons:            {NEURONS}
    Non-neg coeffs:     {non_neg_coeffs}   
    Link functions:''')
    for link_idx in range(len({SELECT_LINK_FXN})):
        print(f'       Node {link_idx + 1} > {SELECT_LINK_FXN[link_idx]}')

    print(f'''          
GD INFO:
    GD method:  {gd_method}
    LR method:  {lr_method}
    Conv method:{conv_method}
    Iterations: {gd_iterations}''')


















