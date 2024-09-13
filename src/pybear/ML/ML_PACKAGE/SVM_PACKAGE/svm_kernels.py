import numpy as n
import sparse_dict as sd


def kernel_function_list():
    return [
            'LINEAR',
            'POLYNOMIAL',
            'GAUSSIAN'
            ]


def linear(DOT_MATRIX, return_as=None):
    '''Linear kernel for list-type objects.  Return as ndarray or sparse dict.'''
    if return_as is None: return_as = 'ARRAY'
    return_as = return_as.upper()

    if return_as == 'SPARSE_DICT': return sd.zip_list(DOT_MATRIX)
    elif return_as == 'ARRAY': return DOT_MATRIX


def sparse_linear(SPARSE_DOT_DICT, return_as=None):
    '''Linear kernel for sparse dicts.  Return as ndarray or sparse dict.'''
    if return_as is None: return_as = 'SPARSE_DICT'
    return_as = return_as.upper()

    if return_as == 'ARRAY': return sd.unzip_to_ndarray(SPARSE_DOT_DICT)[0]
    elif return_as == 'SPARSE_DICT': return SPARSE_DOT_DICT



def polynomial(DOT_MATRIX, constant, exponent, return_as=None):
    '''Polynomial kernel for list-type objects.  Return as ndarray or sparse dict.'''
    if return_as is None: return_as = 'ARRAY'
    return_as = return_as.upper()

    if return_as == 'ARRAY': return (DOT_MATRIX + constant) ** exponent
    elif return_as == 'SPARSE_DICT': return sd.scalar_power(sd.scalar_add(sd.zip_list(DOT_MATRIX), constant), exponent)


def sparse_polynomial(SPARSE_DOT_DICT, constant, exponent, return_as=None):
    '''Polynomial kernel for sparse dicts.  Return as ndarray or sparse dict.'''
    if return_as is None: return_as = 'SPARSE_DICT'
    return_as = return_as.upper()

    if return_as == 'ARRAY': return (sd.unzip_to_ndarray(SPARSE_DOT_DICT)[0] + constant) ** exponent
    elif return_as == 'SPARSE_DICT': return sd.scalar_power(sd.scalar_add(SPARSE_DOT_DICT, constant), exponent)


def gaussian(LIST1, sigma, return_as=None):
    '''Gaussian (rbf) kernel for list-type objects.  Return as ndarray or sparse dict.  Must be [] = rows.'''

    if return_as == None: return_as = 'ARRAY'
    return_as = return_as.upper()

    GAUSSIAN_DOT = n.zeros((len(LIST1), len(LIST1)), dtype=n.float64)
    for list_idx1 in range(len(LIST1)):
        for list_idx2 in range(list_idx1 + 1):    # MUST GET DIAGONAL SO +1
            gaussian_dot = n.sum((LIST1[list_idx1] - LIST1[list_idx2]) ** 2)
            GAUSSIAN_DOT[list_idx1][list_idx2] = gaussian_dot
            GAUSSIAN_DOT[list_idx2][list_idx1] = gaussian_dot

    GAUSSIAN_DOT =  n.exp(-GAUSSIAN_DOT / (2 * sigma**2))

    if return_as == 'SPARSE_DICT':
        GAUSSIAN_DOT = sd.zip_list_as_py_float(GAUSSIAN_DOT)

    return GAUSSIAN_DOT


def sparse_gaussian(DICT1, DICT2, sigma, return_as=None):
    '''Gaussian (rbf) kernel for sparse dicts.  Return as ndarray or sparse dict.'''

    if sd.inner_len(DICT1) != sd.inner_len(DICT2):
        raise Exception(f'svm_kernels.gaussian() requires vectors of equal length for gaussian dot product.')

    if return_as == None: return_as = 'SPARSE_DICT'
    return_as = return_as.upper()

    return sd.core_symmetric_gaussian_dot(DICT1, sigma, return_as=return_as)
































