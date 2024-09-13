import numpy as np
import sparse_dict as sd
from ML_PACKAGE.NN_PACKAGE.link_functions import link_fxns as lf
from general_data_ops import get_shape as gs


# MATRIX MULTIPLICATIONS FORWARD PROP --- CREATE OUTPUT VECTOR AND OTHER OBJECTS NEEDED FOR BACKPROP
# CALLED BY NN

# DATA MUST COME IN AS [[] = COLUMN]

def output_vector_calc_backprop(DATA, ARRAY_OF_NODES, ARRAY_OF_z_x, ARRAY_OF_a_x_b4, ARRAY_OF_a_x_after,
                       SELECT_LINK_FXN, OUTPUT_VECTOR, activation_constant):

    ARRAY_OF_z_x = np.zeros((1, len(ARRAY_OF_z_x)), dtype=object)[0]
    ARRAY_OF_a_x_b4 = np.zeros((1, len(ARRAY_OF_a_x_b4)), dtype=object)[0]
    ARRAY_OF_a_x_after = np.zeros((1, len(ARRAY_OF_a_x_after)), dtype=object)[0]

    INTERMEDIATE_RESULT = DATA.copy()

    for node in range(len(ARRAY_OF_NODES)):
        # APPEND ZEROS TO a_x_b4 TO CREATE a_x_after, RIGHT BEFORE MATMUL INTO NEXT NODE
        if node != 0 and activation_constant != 0:
            #activation_constant IS APPENDED TO INTERMEDIATE RESULT FOR EACH INTERMEDIATE RESULT EXCEPT LAST OUTPUT
            INTERMEDIATE_RESULT = np.insert(INTERMEDIATE_RESULT, len(INTERMEDIATE_RESULT), activation_constant, axis=0)
            # DICT CAN NEVER GET BEYOND NODE 0

        # APPEND THIS AUGMENTED (OR NOT AUGMENTED) MATRIX TO ARRAY_OF_a_x_after
        ARRAY_OF_a_x_after[node] = sd.unzip_to_ndarray(INTERMEDIATE_RESULT)[0] if isinstance(INTERMEDIATE_RESULT, dict) \
            else INTERMEDIATE_RESULT
        # MATMUL a_x_after WITH THE FOLLOWING NODE

        if node == 0:   # DUPLICATED LIST MATMUL LIKE THIS TO BE CLEAR THAT DICT MATMUL CAN ONLY EVER HAPPEN ON NODE 0
            if isinstance(INTERMEDIATE_RESULT, (list, tuple, np.ndarray)):
                INTERMEDIATE_RESULT = np.matmul(ARRAY_OF_NODES[node].astype(np.float64),
                                                INTERMEDIATE_RESULT.astype(np.float64), dtype=np.float64)
            elif isinstance(INTERMEDIATE_RESULT, dict):
                # BEAR MADE CHANGES TO core_hybrid_matmul 12/7/22. NOT SURE IF PREVIOUS RETURN WAS AS 'ROW'.
                # IF NO ISSUE OK TO DELETE THIS NOTE.
                INTERMEDIATE_RESULT = sd.core_hybrid_matmul(ARRAY_OF_NODES[node].astype(np.float64),
                                                            INTERMEDIATE_RESULT, return_as='ARRAY', return_orientation='ROW')
        else:
            INTERMEDIATE_RESULT = np.matmul(ARRAY_OF_NODES[node].astype(np.float64),
                                                INTERMEDIATE_RESULT.astype(np.float64), dtype=np.float64)

        # APPEND RAW MATMUL OUTPUT TO ARRAY_OF_zs
        ARRAY_OF_z_x[node] = INTERMEDIATE_RESULT
        # PASS z THROUGH LINK, TURNING IT INTO a_x
        INTERMEDIATE_RESULT = lf.link_fxns(INTERMEDIATE_RESULT, SELECT_LINK_FXN[node])
        # APPEND OUTPUT FROM LINK TO ARRAY_OF_a_x_b4
        ARRAY_OF_a_x_b4[node] = INTERMEDIATE_RESULT

        # B4 MEANS B4 APPENDING 1s, AFTER MEANS AFTER APPENDING 1s (OR NOT APPENDED IF SO CHOSEN)
        '''ARRAYS SHOULD LOOK LIKE THIS:  (USING 4 NODES JUST AS AN EXAMPLE)
        ARRAY_OF_a_x_after = [] ---> [ DATA           , a0after            , a1after           , a2after                    ]
        ARRAY OF NODES --->          [     NODE0      ,        NODE1       ,        NODE2      ,        NODE3               ]
        ARRAY_OF_z_x = [] --->       [          z0    ,             z1     ,             z2    ,             z3             ]
        ARRAY_OF_a_x_b4 = [] --->    [            a0b4,               a1b4 ,               a2b4,                a3b4(OUTPUT)]
        '''

    ########################################################################################################
    # MULTI-OUT LINK IS A PSEUDO-LINK FXN THAT TRANSPOSES MULTI-OUT OUTPUT INTO CORRECT ORIENTATION
    # OUTPUT IS BUILT AS
    '''
    [                                                 [
        [a, b, c, d, e],                              [a,a,a],
        [a, b, c, d, e],    THEN IS TRANSPOSED TO     [b,b,b],
        [a, b, c, d, e]                               etc.
    ]                                                 ]
    
    WHERE [a,a,a] ETC. IS THE CORRECT TARGET VECTOR ORIENTATION
    MULTI-OUT IS CALCULATED SIMILAR TO THE WAY SOFTMAX IS HANDLED, THEN REORIENTED.
    '''

    if SELECT_LINK_FXN[-1].upper() == 'MULTI-OUT':
        INTERMEDIATE_RESULT = lf.link_fxns(INTERMEDIATE_RESULT, 'Multi-out')
    ########################################################################################################

    OUTPUT_VECTOR = INTERMEDIATE_RESULT.copy()

    return OUTPUT_VECTOR, ARRAY_OF_z_x, ARRAY_OF_a_x_b4, ARRAY_OF_a_x_after





def output_vector_calc(DATA, ARRAY_OF_NODES, SELECT_LINK_FXN, OUTPUT_VECTOR, activation_constant):

    INTERMEDIATE_RESULT = DATA.copy()

    for node in range(len(ARRAY_OF_NODES)):

        if node != 0 and activation_constant != 0:
            INTERMEDIATE_RESULT = np.insert(INTERMEDIATE_RESULT, len(INTERMEDIATE_RESULT), activation_constant, axis=0)


        if isinstance(INTERMEDIATE_RESULT, (list, tuple, np.ndarray)):
            INTERMEDIATE_RESULT = np.matmul(ARRAY_OF_NODES[node].astype(np.float64),
                                        INTERMEDIATE_RESULT.astype(np.float64), dtype=np.float64)

        elif isinstance(INTERMEDIATE_RESULT, dict):
            # BEAR MADE CHANGES TO core_hybrid_matmul 12/7/22. NOT SURE IF PREVIOUS RETURN WAS AS 'ROW'.
            # IF NO ISSUE OK TO DELETE THIS NOTE.
            INTERMEDIATE_RESULT = sd.core_hybrid_matmul(ARRAY_OF_NODES[node].astype(np.float64),
                                        INTERMEDIATE_RESULT.astype(np.float64), return_as='ARRAY', return_orientation='ROW')

        INTERMEDIATE_RESULT = lf.link_fxns(INTERMEDIATE_RESULT, SELECT_LINK_FXN[node])

    if SELECT_LINK_FXN[-1].upper() == 'MULTI-OUT':
        INTERMEDIATE_RESULT = lf.link_fxns(INTERMEDIATE_RESULT, 'Multi-out')
    ########################################################################################################

    OUTPUT_VECTOR = INTERMEDIATE_RESULT

    return OUTPUT_VECTOR



if __name__ == '__main__':

    DATA = np.random.randint(1,10,(3,5))
    ARRAY_OF_NODES_COPY = np.random.rand(3,3)
    ARRAY_OF_NODES = ARRAY_OF_NODES_COPY.copy()
    # ARRAY_OF_NODES.resize(1,3)
    print(ARRAY_OF_NODES)
    print(np.matmul(ARRAY_OF_NODES, DATA))
    SELECT_LINK_FXN = ['Softmax']
    OUTPUT_VECTOR = []
    activation_constant = 0

    OUTPUT = output_vector_calc(DATA, ARRAY_OF_NODES, SELECT_LINK_FXN, OUTPUT_VECTOR, activation_constant)
    print(f'\nOUTPUT VECTOR:')
    print(OUTPUT)





