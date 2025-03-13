import numpy as n

#CALLED BY shmn_inverse_check, NN
def n_inverse_exception(ARRAY, name, Print):

    try:
        ARRAY1_INV = n.linalg.inv(ARRAY)

        if Print == 'Y':
            print(f'\n{name} inverse exists.')
            sign_, logdet_ = n.linalg.slogdet(ARRAY1_INV)
            print(f'Determinant = {sign_ * n.exp(logdet_)}')
            print(f'Min element = {n.min(ARRAY1_INV)}')
            print(f'Max element = {n.max(ARRAY1_INV)}\n')

        return ARRAY1_INV

    except:
        print('')
        if n.linalg.LinAlgError:
            print(f'\n{name} has numpy.linalg.LinAlgError, singular matrix\n')
        else:
            print(f'\nCannot invert {name} for error other than numpy.linalg.LinAlgError\n')

        return []




