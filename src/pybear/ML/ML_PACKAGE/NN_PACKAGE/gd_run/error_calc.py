import numpy as n
from general_data_ops import numpy_math as nm
from general_list_ops import list_of_lists_merger as llm

class NS(nm.NumpyMath): pass

#*******ERROR CALCULATION*******************************************************************************
#AKA COST FUNCTION

def cost_functions():
    return {'s': 'least-squares',
            'l': 'minus log-likelihood',
            'u': 'unsupervised_binary'
    }



#CALLED BY NNCoreConfigCode, NN
def error_calc(ARRAY_OF_NODES, TARGET_VECTOR, OUTPUT_VECTOR, cost_fxn, new_error_start,
               SELECT_LINK_FXN, rglztn_type, rglztn_fctr):

    cost_fxn = cost_fxn.upper()

    #LEAST SQUARES
    if cost_fxn in 'S':
        # FOR EVERYTHING BUT SOFTMAX
        if 'SOFTMAX' not in SELECT_LINK_FXN[-1].upper():
            new_error_start = NS().sumf(n.power(NS().subtractf(n.array(TARGET_VECTOR), OUTPUT_VECTOR), 2))

        # IF SOFTMAX...
        else:
            new_error_start = NS().sumf(n.power(NS().subtractf(TARGET_VECTOR, OUTPUT_VECTOR), 2))

        # total_error += (y - h_x)**2


    # -LOG LIKELIHOOD
    elif cost_fxn == 'L':

        if SELECT_LINK_FXN[-1].upper() == 'LOGISTIC':

            new_error_start = NS().sumf(
                        -NS().multiplyf(n.array(TARGET_VECTOR), NS().logf(OUTPUT_VECTOR)) \
                        - NS().multiplyf(NS().subtractf(1, n.array(TARGET_VECTOR)), NS().logf(NS().subtractf(1, OUTPUT_VECTOR)))
                )

        elif SELECT_LINK_FXN[-1].upper() == 'SOFTMAX':
            #WORRY ABOUT ONLY THOSE SOFTMAXES WHERE TARGET IS 1
            CALC1 = [x for x in llm.list_of_lists_merger(NS().multiplyf(OUTPUT_VECTOR, TARGET_VECTOR)) if x != 0]
            new_error_start = n.sum([-NS().logf(CALC1)])
        else:
            raise ValueError(f'-Log likelihood not defined for final link "{SELECT_LINK_FXN[-1]}".')

    # UNSUPERVISED BINARY
    elif cost_fxn == 'U':
        # 9-12-2021 I THINK THAT THIS COULD (OR MAYBE EVEN IS LIKELY TO WITH RGLZTN) CONVERGE TO ALL ZEROS FOR OUTPUT
        # SO PUTTING A LENGTH REWARD FOR HAVING LENGTH IN FINAL OUTPUT

        # NOT DONE
        new_error_start = NS().sumf(NS().multiplyf(NS().subtractf(1, 2 * OUTPUT_VECTOR))) - \
                      0.5 * 1 * n.sum([n.sum(n.power(X, 2)) for X in OUTPUT_VECTOR])

    else:
        raise NotImplementedError(f'\n*** UNKNOWN cost_fxn "{cost_fxn}" IN error_calc.error_calc(). ***')


    # ADD RGLZTN WEIGHT TO new_error_start
    if rglztn_type == 'L1':
        new_error_start += rglztn_fctr * n.sum([n.sum(n.abs(X)) for X in ARRAY_OF_NODES])

    elif rglztn_type == 'L2':
        new_error_start += 0.5 * rglztn_fctr * n.sum([n.sum(n.power(X, 2)) for X in ARRAY_OF_NODES])


    return new_error_start





if __name__ == '__main__':
    pass





