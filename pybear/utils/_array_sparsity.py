# Authors:
#
#       Bill Sousa
#
# License: BSD 3 clause


import numpy as np


def array_sparsity(a):
    """Calculate the sparsity (percentage of zeros) of an array-like.

    Parameters
    ---------
    a: array-like - object for which to calculate sparsity. Must be able to
                    convert to a numpy.ndarray and must not be empty.


    Returns
    ------
    sparsity: float - percentage of zeros in a.


    See Also
    -------
    pybear.sparse.utils.sparsity for a sparse dictionary implementation


    Notes
    ----
    None


    Example
    ------
    >>> import numpy as np
    >>> from pybear.utils import array_sparsity
    >>> a = np.random.randint(0,10,(1000,500), dtype=np.uint8)
    >>> sparsity = array_sparsity(a)
    >>> print(sparsity)
        9.949

    """


    err_msg = (f"'a' must be a non-empty array-like that can be converted to a "
               f"numpy.ndarray.")


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


    return (a == 0).astype(np.int32).sum() / a.size * 100














