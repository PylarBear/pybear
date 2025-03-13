import numpy as n

#CALLED BY NN
def generate_hessian(BATCH_MATRIX, BATCH_TARGET_VECTOR, h_x, SMAX_h_X, select_link_fxn):

    # LINK_FXNS = ['ReLU_lower',
    #              'ReLU_upper',
    #              'ReLU_lower_and_upper',
    #              'Logistic',
    #              'Tanh',
    #              'Softmax',
    #              'None']

    if select_link_fxn in ['None','ReLU_lower','ReLU_upper','ReLU_lower_and_upper']:
        H = n.matmul(n.transpose(BATCH_MATRIX), BATCH_MATRIX, dtype=float)

    elif select_link_fxn in ['Logistic','Softmax']:

        H = []    #H HOLD THE HESSIANS ASSOCIATED W EACH LABEL
        for label_H_idx in range(len(BATCH_TARGET_VECTOR)):    #CREATE & FILL W PLACEHOLDERS FOR EASY OVERWRITE LATER
            H.append([])
            for vector1 in BATCH_MATRIX:
                H[-1].append([])
                for vector2 in BATCH_MATRIX:
                    H[-1][-1].append(float(0))   #FILL MATRIX W ZEROS AS PLACEHOLDERS

            for vector1_idx in range(len(BATCH_MATRIX)):
                for vector2_idx in range(vector1_idx + 1):
                    vector_totals = 0
                    for example in range(len(BATCH_MATRIX[0])):
                        if len(BATCH_TARGET_VECTOR) == 1:   #LOGISTIC
                            vector_totals += BATCH_MATRIX[vector1_idx][example] * BATCH_MATRIX[vector2_idx][example] \
                                             * h_x[example] * (1 - h_x[example])
                        elif len(BATCH_TARGET_VECTOR) > 1:   #SOFTMAX
                            if BATCH_TARGET_VECTOR[label_H_idx][example] == 1:
                                vector_totals += BATCH_MATRIX[vector1_idx][example] * BATCH_MATRIX[vector2_idx][example] \
                                                 * h_x[example] * (1 - h_x[example])
                            # elif BATCH_TARGET_VECTOR[label_H_idx][example] != 1:
                            #     vector_totals += -BATCH_MATRIX[vector1_idx][example] * BATCH_MATRIX[vector2_idx][example] \
                            #                      * SMAX_h_X[label_H_idx][example] * h_x[example]

                    H[label_H_idx][vector1_idx][vector2_idx] = vector_totals    #H IS SYMMETRIC
                    H[label_H_idx][vector2_idx][vector1_idx] = vector_totals



    elif select_link_fxn == 'Tanh':
        pass


    return H