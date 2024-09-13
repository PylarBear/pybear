import numpy as n, warnings, sys
from copy import deepcopy
from ML_PACKAGE.NN_PACKAGE.gd_run import gd_gd as gdgd, gd_rmsprop as gdr


def gd_adam(NEW_GRADIENT, OLD_GRADIENT, OLD_RMS, momentum_weight, iteration):
    '''
    ADAM =
    INITIALIZE VdW=0, SdW=0
    m_factor = momentum weight  ( TYPICALLY 0.9 )
    beta = RMSProp weight ( ADAM PPL RECOMMEND .999 )
    alpha = learning rate
    ON ITERATION t:
	1) COMPUTE dW (GRADIENT) ON CURRENT MINI-BATCH
	2) COMPUTE MOMENTUM WEIGHTED GRADIENT > VdW = m_factor * VdW + (1-m_factor) * dW
	3) COMPUTE RMSPROP > SdW = beta * SdW + (1-beta)*dW^2  (element-wise square)
	4) COMPUTE
		BIAS CORRECTION FOR VdW (MOMENTUM WEIGHTED GRADIENT)
		VdW_corr = VdW / (1-m_factor^t)
		BIAS CORRECTION FOR SdW
		SdW_corr = SdW / (1-beta^t)
	5) COMPUTE
		W := W - alpha * VdW_corr / [ sqrt(SdW_corr) + epsilon ]     epsilon to prevent overflow if SdW near zero USUALLY 1E-8
    '''

    beta = .999
    epsilon = 1e-8

    GD_GRADIENT = gdgd.gd_gd(NEW_GRADIENT, OLD_GRADIENT, momentum_weight)
    NEW_RMS = gdr.gd_rmsprop(NEW_GRADIENT, OLD_RMS)[1]

    GD_GRADIENT_corr = GD_GRADIENT / (1 - momentum_weight ** iteration)
    NEW_RMS_corr = NEW_RMS / (1 - beta ** iteration)

    ADAM_GRADIENT = GD_GRADIENT_corr / (n.sqrt(NEW_RMS_corr) + epsilon)

    return ADAM_GRADIENT, GD_GRADIENT, NEW_RMS













