import numpy as n
from copy import deepcopy


def gd_rmsprop(NEW_GRADIENT, OLD_RMS):
    '''
    RMSPROP =
    INITIALIZE VdW=0, SdW=0
    beta = RMSProp weight
    alpha = learning rate
    ON ITERATION t:
    1) COMPUTE dW (GRADIENT) ON CURRENT MINI-BATCH
    2) COMPUTE SdW = beta*SdW + (1-beta)*dW^2 (ELEMENTWISE SQUARE)     (beta is not momentum weight, it is unique to RMSProp)
    3) COMPUTE W := W - alpha * dW / [ sqrt(SdW) + epsilon ]     epsilon to prevent overflow if SdW near zero, USUALLY 1E-8
    '''

    beta = .999
    epsilon = 1e-8

    NEW_RMS = beta * OLD_RMS + (1-beta) * n.power(NEW_GRADIENT, 2)

    RMS_GRADIENT = NEW_GRADIENT / (n.sqrt(NEW_RMS) + epsilon)

    return RMS_GRADIENT, NEW_RMS





