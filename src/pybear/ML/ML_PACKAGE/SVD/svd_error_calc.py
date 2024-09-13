import numpy as n


def svd_error_calc(OUTPUT_VECTOR, TARGET_VECTOR):

    new_error_start = n.sum(n.power(TARGET_VECTOR - OUTPUT_VECTOR, 2))

    return new_error_start













