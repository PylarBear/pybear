# Authors:
#
#       Bill Sousa
#
# License: BSD 3 clause


import numpy as np


def permuter(vector_of_vectors) -> list:

    """Given a vector of length n that contains n vectors of unique values with
    lengths (l1, l2,...ln), and whose total possible number of unique combinations
    is pn, generate an array of shape (pn, n) that contains in its rows the index
    positions given by permuting through all possible unique combinations
    of values within those vectors.

    Parameters
    ----------
    vector_of_vectors:
        array-like - vector of vectors of non-zero length

    Return
    ------
    -
        permutations : list of shape (number of possible combinations,
                        number of vectors)

    See Also
    --------
    itertools.product
        for another implementation that returns values instead of indices.

    Example
    -------
    >>> from pybear.utils import permuter
    >>> vector1 = ['a', 'b', 'c']
    >>> vector2 = ['w', 'x']
    >>> vector3 = ['y', 'z']
    
    >>> vector_of_vectors = [vector1, vector2, vector3]

    >>> output = permuter(vector_of_vectors)

    output:
    [[0,0,0],
     [0,0,1],
     [0,1,0],
     [0,1,1],
     [1,0,0],
     [1,0,1],
     [1,1,0],
     [1,1,1],
     [2,0,0],
     [2,0,1],
     [2,1,0],     
     [2,1,1]]

    """


    cp_vector_of_lens = np.array(list(map(len, vector_of_vectors)))

    if (cp_vector_of_lens <= 0).any():
        raise ValueError(f"vector_of_vectors cannot contain any empty vectors")
        
    
    def recursive_fxn(cp_vector_of_lens):
        if len(cp_vector_of_lens)==1:
            seed_array = np.zeros((cp_vector_of_lens[0],
                                   len(vector_of_vectors)),
                                  dtype=int
            )
            seed_array[:, -1] = range(cp_vector_of_lens[0])
            return seed_array
        else:
            seed_array = recursive_fxn(cp_vector_of_lens[1:])
            stack = np.empty((0, len(vector_of_vectors)), dtype=np.uint32)
            for param_idx in range(cp_vector_of_lens[0]):
                filled_array = seed_array.copy()
                col_idx = len(vector_of_vectors) - len(cp_vector_of_lens)
                filled_array[:, col_idx] = param_idx
                del col_idx
                stack = np.vstack((stack, filled_array))
    
            del filled_array
            return stack

    permutations = list(map(tuple, recursive_fxn(cp_vector_of_lens)))

    del cp_vector_of_lens, recursive_fxn
        
    return permutations










