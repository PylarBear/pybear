# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import warnings

import scipy.sparse as ss



def scipy_sparse_preslice_handle(X: any) -> any:

    """
    Check if a data container's columns can be extracted by index slicing.
    This is only expected to be an issue with scipy sparse objects.

    Summary of scipy sparse 'getcol' method and slicing:

    version 1.13
    All matrix objects have 'getcol' method
    All array objects have 'getcol' method
    bsr matrix/array, coo matrix/array, dia matrix/array cannot slice
    all other matrix/array objects can slice columns by X[:, [col_idxs]]

    versions 1.14 - 1.15 (most recent as of writing)
    ALL ARRAY OBJECTS DO NOT HAVE 'getcol' METHOD
    All matrix objects have 'getcol' method
    bsr matrix/array, coo matrix/array, dia matrix/array cannot slice
    all other matrix/array objects can slice columns by X[:, [col_idxs]]

    pybear needs a one-size-fits-all solution for extracting single or
    multiple columns from scipy sparse objects. Only matrix objects have
    a 'getcol' method, so arrays are excluded. Column slicing is the
    only other means available, but is not available for coo, dia, and
    bsr. (Which begs the question, how to extract columns from coo, dia,
    and bsr directly?) The pybear solution is to use column slicing, and
    to force coo, dia, and bsr over to csc with a copy, and persist the
    csc in memory for the entire session (and therefore all column
    extractions.) The alternative (higher cost) solution would be to map
    the scipy objects to csc_array, extract the column, and map back to
    the original container for every single column extraction.


    Parameters
    ----------
    X:
        any - the object to be checked if it is a scipy bsr, coo, or dia
        matrix / array, and if so, mapped over to csc_array format.


    Return
    ------
    X:
        any - if the original object was scipy bsr, coo, or dia, the
        object converted to csc array. otherwise, the original object.

    """


    # ss sparse that cant be sliced
    if isinstance(X,
        (ss.coo_matrix, ss.dia_matrix, ss.bsr_matrix,
         ss.coo_array, ss.dia_array, ss.bsr_array)
    ):
        warnings.warn(
            f"pybear works hard to avoid mutating or creating copies of "
            f"your original data. \nyou have passed your data as {type(X)}, "
            f"which cannot be sliced by columns. pybear needs to create "
            f"a copy. \nto avoid this, pass your sparse data as csr, csc, "
            f"lil, or dok."
        )
        _X = ss.csc_array(X.copy())
    else:
        _X = X


    return _X







