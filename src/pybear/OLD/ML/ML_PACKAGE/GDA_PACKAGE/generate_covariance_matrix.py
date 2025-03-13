import numpy as n




def generate_covariance_matrix(DATA):

    NUMPY = n.array(DATA, dtype=float)
    COVARIANCE_MATRIX = n.divide(
                                    n.matmul(NUMPY, NUMPY.transpose(), dtype=float
                                    ),
                                len(NUMPY[0])
    )

    return COVARIANCE_MATRIX


