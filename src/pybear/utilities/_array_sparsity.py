# Authors:
#
#       Bill Sousa
#
# License: BSD 3 clause


import numpy as np


def array_sparsity(a) -> float:

    """
    Calculate the sparsity (percentage of zeros) of an array-like.


    Parameters
    -----------
    a:
        array-like - object for which to calculate sparsity. Must be
            able to convert to a numpy.ndarray and must not be empty.


    Return
    ------
    return
        sparsity: float - percentage of zeros in a.


    See Also
    --------
    pybear.sparse_dict.utils.sparsity for a sparse dictionary implement-
    ation


    Examples
    --------
    >>> import numpy as np
    >>> from pybear.utilities import array_sparsity
    >>> a = np.array([[0,1,0,2,0],[1,0,2,0,3]])
    >>> array_sparsity(a)
    50.0

    """


    err_msg = (f"'a' must be a non-empty array-like that can be converted "
               f"to a numpy.ndarray.")


    if isinstance(a, (str, type(None), dict)):
        raise TypeError(err_msg)

    try:
        a = np.array(list(a))
        if a.size == 0:
            raise ValueError(err_msg)
    except ValueError:
        raise
    except:
        raise TypeError(err_msg)


    return float((a == 0).astype(np.int32).sum() / a.size * 100)














