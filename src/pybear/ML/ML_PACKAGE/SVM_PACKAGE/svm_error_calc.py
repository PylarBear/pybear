import numpy as np
from general_data_ops import numpy_math as nm
from debug import IdentifyObjectAndPrint as ioap
from ML_PACKAGE.SVM_PACKAGE import svm_output_calc as svmoc, link_fxns as lf


class NS(nm.NumpyMath): pass


#*******ERROR CALCULATION*******************************************************************************
#AKA COST FUNCTION

def cost_functions():
    return {
            'H': 'HINGE LOSS',
            'C': 'COUNT WRONG',
            'P': 'PERCENT WRONG'
            }


def svm_error_calc(TARGET, OUTPUT, cost_fxn, new_error_start):

    # HINGE LOSS
    if cost_fxn == 'H':
        # L(y) = max(0, 1 - t * y),   t == ground truth, y == actual output mx+b, not {-1, 1})
        '''
        # KEEP THIS HERE IN CASE DONT HAVE OUTPUT YET
        if OUTPUT is None:
            OUTPUT = svmoc.svm_output_calc(DATA, K, ALPHAS, b)
        '''

        new_error_start = np.sum(list(map(np.max, zip([0 for _ in OUTPUT[0]], 1 - TARGET[0] * OUTPUT[0]))))


    # COUNT WRONG
    elif cost_fxn == 'C':

        # CONVERT TO SVM_Perceptron
        OUTPUT = lf.link_fxns(OUTPUT[0], 'SVM_Perceptron')

        new_error_start = np.sum(TARGET[0] != OUTPUT[0])


    # PERCENT WRONG
    elif cost_fxn == 'P':

        # CONVERT TO SVM_Perceptron
        OUTPUT = lf.link_fxns(OUTPUT[0], 'SVM_Perceptron')

        new_error_start = 100 * np.sum(TARGET[0] != OUTPUT[0]) / len(TARGET[0])

    else:
        raise Exception(f'*** svm_error_calc() >>> INVALID cost_fxn ({cost_fxn}) ***')

    return new_error_start





if __name__ == '__main__':
    pass





