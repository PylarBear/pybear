import numpy as n
from general_data_ops import numpy_math as nm

#CALLED BY NN
#IF ADDING A NEW LINK FUNCTION: 1) PUT IN define_links 2) IF IS MULTI-OUT, PUT IN define_multi_out_links
# 3) PUT LINK FUNCTION IN link_fxns 4) PUT LINK DERIVATIVE IN link_derivatives

class NS(nm.NumpyMath):
    pass


def define_links():
    return ['ReLU_lower',
                 'Leaky_ReLU_lower',
                 'ReLU_upper',
                 'ReLU_lower_and_upper',
                 'Logistic',
                 'Perceptron',
                 'SVM_Perceptron',
                 'Tanh',
                 'Softmax',
                 'Multi-out',
                 'None'
    ]

def define_multi_out_links():
    return ['Softmax',
            'Multi-out'
    ]

def define_single_out_links():
    return [x for x in define_links() if x not in define_multi_out_links()]


#CALLED BY NNCoreRunCode, output_vector_calc
def link_fxns(ARG, select_link_fxn):

    # ARG is a numpy array aka [[ ]]

    #None
    if select_link_fxn in ['None']:
        pass

    #ReLU_lower
    elif select_link_fxn in ['ReLU_lower']:
        ARG = ARG * (ARG >= 0)

    #Leaky_ReLU_lower
    elif select_link_fxn in ['Leaky_ReLU_lower']:
        ARG = 0.01 * ARG * (ARG < 0) + ARG * (ARG >= 0)

    #ReLU_upper
    elif select_link_fxn in ['ReLU_upper']:
        ARG = ARG * (ARG < 1) + (ARG >= 1)

    #ReLU_lower_and_upper
    elif select_link_fxn in ['ReLU_lower_and_upper']:
        ARG  = ARG * (ARG > 0)
        ARG = ARG * (ARG < 1) + (ARG >= 1)

    #Logistic
    elif select_link_fxn == 'Logistic':
        ARG = list(map(list, ARG))  # BANDAID TO GET THIS TO WORK 7-2-22
        ARG = 1 / (1 + n.exp(-n.array(ARG)))

    #Tanh
    elif select_link_fxn == 'Tanh':
        ARG = n.tanh(n.array(ARG))


    elif select_link_fxn == 'Perceptron':
        ARG = 1 * (ARG >= 0)

    elif select_link_fxn == 'SVM_Perceptron':
        ARG = 2 * (ARG >= 0) - 1

    elif select_link_fxn == 'Softmax':
        # TRANSPOSE ARG SO THAT ASSOCIATED PROBABILITIES GO IN ONE LIST (#LISTS = #EXAMPLES),
        # WHICH IS THEN RUN THRU THE LINK FXN TO NORMALIZE SUM TO ZERO
        # ONCE THE LINK IS CALCULATED, TRANSPOSE ARG BACK TO ORIGINAL (#LABELS = #LISTS)
        # HAVE TO DO IT OUT THIS WAY BECAUSE ONE-LINER PY LIST GENERATOR METHOD BLEW UP

        ARG = n.transpose(ARG)

        for example_idx in range(len(ARG)):
                ARG[example_idx] = NS().dividef(n.exp(ARG[example_idx], dtype=float),
                                        NS().sumf(n.exp(ARG[example_idx]))
                )

        ARG = n.transpose(ARG)


    elif select_link_fxn == 'Multi-out':
        ARG = n.transpose(ARG)

    else:
        raise NotImplementedError(f'link_fxns.linkfxns(): There is no formula for "{select_link_fxn}" link.')

    return ARG


def link_derivative(INTERMEDIATE, select_link_fxn):
    if select_link_fxn == 'Logistic':
        return NS().multiplyf(INTERMEDIATE, NS().subtractf(1, INTERMEDIATE))

    elif select_link_fxn == 'Softmax':
        return NS().multiplyf(INTERMEDIATE, NS().subtractf(1, INTERMEDIATE))

    elif select_link_fxn == 'ReLU_lower':
        return (INTERMEDIATE >= 0)**2   # SQUARING TO TURN BOOLEANS INTO 0,1

    elif select_link_fxn == 'Leaky_ReLU_lower':
        return 0.01 * (INTERMEDIATE < 0) + (INTERMEDIATE >= 0)

    elif select_link_fxn == 'ReLU_upper':
        return (INTERMEDIATE <= 1)**2   # SQUARING TO TURN BOOLEANS INTO 0,1

    elif select_link_fxn == 'ReLU_lower_and_upper':
        return (INTERMEDIATE >= 0) * (INTERMEDIATE <= 1)

    elif select_link_fxn == 'Perceptron':
        return INTERMEDIATE * 0

    elif select_link_fxn == 'SVM_Perceptron':
        return INTERMEDIATE * 0

    elif select_link_fxn == 'Tanh':
        return NS().subtractf(1, n.tanh(INTERMEDIATE)**2)

    elif select_link_fxn == 'None':
        return INTERMEDIATE / INTERMEDIATE    # ALL ONES
    





if __name__ == '__main__':
    pass



