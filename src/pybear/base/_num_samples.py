# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#




def num_samples(OBJECT) -> int:

    """
    Return the number of samples in an array-like OBJECT.

    OBJECT must have a 'shape' attribute.

    numpy & pandas: OBJECT must be 1 or 2 dimensional.
    scipy: OBJECT must be 2 dimensional.
    If OBJECT is a 1D vector (i.e., len(shape)==1), return len(OBJECT).


    Parameters
    ----------
    OBJECT:
        array-like that has a 'shape' attribute - object to find the
        number of samples in.


    Return
    ------
    -
        rows: int - Number of samples.

    """

    try:
        OBJECT.shape
    except:
        raise ValueError(
            f"\nThe passed object does not have a 'shape' attribute. \nAll "
            f"pybear estimators and transformers require data-bearing objects "
            f"to have a 'shape' attribute, like numpy array, pandas dataframes, "
            f"and scipy sparse matrices / arrays."
        )


    if hasattr(OBJECT, 'toarray') and len(OBJECT.shape) != 2:  # is scipy sparse
        # there is inconsistent behavior with scipy array/matrix and 1D.
        # in some cases scipy.csr_array is allowing 1D to be passed and
        # in other cases it is not. scipy.csr_matrix takes a 1D and reports
        # 2D shape. avoid the whole issue, force all scipy to be 2D.
        raise ValueError(
            f"pybear requires all scipy sparse objects be 2 dimensional"
        )


    if len(OBJECT.shape) == 1:
        return len(OBJECT)
    elif len(OBJECT.shape) == 2:
        return OBJECT.shape[0]
    else:
        raise ValueError(
            f"The passed object has {len(OBJECT.shape)} dimensions. pybear "
            f"requires that all data-bearing objects be 1 or 2 dimensional."
        )








