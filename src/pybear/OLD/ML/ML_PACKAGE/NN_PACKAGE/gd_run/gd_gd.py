from copy import deepcopy



def gd_gd(NEW_GRADIENT, OLD_GRADIENT, momentum_weight):
    ''' NEW_GRADIENT is calculated gradient during the iteration, OLD_GRADIENT is stored old gradient.'''

    GD_GRADIENT = momentum_weight * deepcopy(OLD_GRADIENT) + (1 - momentum_weight) * deepcopy(NEW_GRADIENT)

    return GD_GRADIENT




