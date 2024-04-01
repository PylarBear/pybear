import numpy as np


def permuter(vector_of_lens):
    """Given an array-like of length L that contains the lengths (l1, l2,...ln) of vectors 
    comprised of unique values, and whose total possible number of combinations is pn, 
    generate an array of shape (pn, L) that contains in its rows the index positions that 
    permute through all possible unique combinations of values within those vectors.
    
    
    Parameters:
    -----------
    vector_of_lens : array-like - vector of positive integers
    
    Returns:
    --------
    permutations : ndarray of shape (number possible combinations, number of lengths)
    
    
    Example:
    --------
    vector1 = ['a', 'b', 'c']
    vector2 = ['w', 'x']
    vector3 = ['y', 'z']
    
    vector_of_lens = [len(vector1), len(vector2)]
    
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
    
    cp_vector_of_lens = np.array(vector_of_lens).copy()

    if (cp_vector_of_lens <= 0).any():
        raise ValueError(f"vector_of_lens cannot contain any <= 0")
        
    
    def recursive_fxn(cp_vector_of_lens):
        if len(cp_vector_of_lens)==1:
            seed_array = np.zeros((cp_vector_of_lens[0], len(vector_of_lens)), dtype=int)
            seed_array[:, -1] = range(cp_vector_of_lens[0])
            return seed_array
        else:
            seed_array = recursive_fxn(cp_vector_of_lens[1:])
            stack = np.empty((0, len(vector_of_lens)), dtype=np.uint32)
            for param_idx in range(cp_vector_of_lens[0]):
                filled_array = seed_array.copy()
                filled_array[:, len(vector_of_lens) - len(cp_vector_of_lens)] = param_idx
                stack = np.vstack((stack, filled_array))
    
            del filled_array
            return stack

    permutations = list(map(tuple, recursive_fxn(cp_vector_of_lens)))

    del cp_vector_of_lens, recursive_fxn
        
    return permutations

permuter([2,2,3])
