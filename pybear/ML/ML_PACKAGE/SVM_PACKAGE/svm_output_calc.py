import numpy as n
from ML_PACKAGE.SVM_PACKAGE import svm_kernels as sk


def svm_train_output_calc(TARGET, K, ALPHAS, b):


    try:
        OUTPUT = n.matmul(ALPHAS * TARGET, K) + b
        # OUTPUT = n.matmul(ALPHAS.astype(float) * TARGET, K.astype(float), dtype=float) + b
        return OUTPUT

    except:
        mismatch_txt = lambda OBJ1, OBJ2, obj_name1, obj_name2: \
            f'*** CANNOT RUN SVMRun.svm_train_output_calc() DUE TO MISMATCH ' + \
                                         f'IN SIZE BETWEEN {obj_name1} ({len(OBJ1)}) AND {obj_name2} ({len(OBJ2)}) ***'
        if len(TARGET) != len(K): mismatch_txt(TARGET, K, 'TARGET', 'K')
        if len(TARGET) != len(ALPHAS): mismatch_txt(TARGET, ALPHAS, 'TARGET', 'ALPHAS')
        if len(K) != len(ALPHAS): mismatch_txt(K, ALPHAS, 'K', 'ALPHAS')
        return [['NA' for _ in range(len(ALPHAS))]]



def svm_dev_test_output_calc(SUPPORT_VECTORS, SUPPORT_TARGETS, SUPPORT_ALPHAS, b, DEV_TEST_DATA, kernel_fxn, constant,
                             exponent, sigma):
    # NOT USING sparse_dict HERE, NON_ZERO_ALPHAS SHOULD ALWAYS BE SMALL

    DOT_MATRIX = n.matmul(n.array(SUPPORT_VECTORS).astype(float), DEV_TEST_DATA.astype(float), dtype=object)

    if kernel_fxn == 'LINEAR':
        K = n.array(sk.linear(DOT_MATRIX), dtype=n.float64)
    elif kernel_fxn == 'POLYNOMIAL':
        K = n.array(sk.polynomial(DOT_MATRIX, constant, exponent), dtype=n.float64)
    elif kernel_fxn == 'GAUSSIAN':
        K = n.array(sk.gaussian(DOT_MATRIX, sigma), dtype=n.float64)
    else:
        raise ValueError(f'\nINVALID kernel_fxn ({kernel_fxn}) IN svm_output_calc.svm_dev_test_output_calc()')

    OUTPUT = n.matmul(n.multiply(SUPPORT_TARGETS, SUPPORT_ALPHAS, dtype=float), K.astype(float), dtype=float) + b

    return n.array([OUTPUT], dtype=n.float64)










