import numpy as n



def gaussian_coefficient(COVARIANCE_MATRIX, dimensions):
    sign_, logdet_ = n.linalg.slogdet(COVARIANCE_MATRIX)
    return 1 / ((2 * n.pi) ** (0.5 * dimensions) * n.sqrt(sign_ * n.exp(logdet_)))


