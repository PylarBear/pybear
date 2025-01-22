# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#




def num_features(OBJECT) -> int:

    """
    Return the number of features in an array-like OBJECT.

    OBJECT must have a 'shape' attribute.

    numpy & pandas: OBJECT must be 1 or 2 dimensional.
    scipy: OBJECT must be 2 dimensional.
    If OBJECT is a 1D vector (i.e., len(shape)==1), return 1.


    Parameters
    ----------
    OBJECT:
        array-like of shape (n_samples, n_features) or (n_samples,) that
        has a 'shape' attribute - object to find the number of features
        in.


    Return
    ------
    -
        features: int - Number of features.

    """

    try:
        OBJECT.shape
    except:
        raise ValueError(
            f"\nThe passed object does not have a 'shape' attribute. "
            f"\nAll pybear estimators and transformers require data-bearing "
            f"objects to have a 'shape' attribute, like numpy arrays, pandas "
            f"dataframes, and scipy sparse matrices / arrays."
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
        return 1
    elif len(OBJECT.shape) == 2:
        return int(OBJECT.shape[1])
    else:
        raise ValueError(
            f"The passed object has {len(OBJECT.shape)} dimensions. pybear "
            f"requires that all data-bearing objects be 1 or 2 dimensional."
        )



    # keep this for reference
    # from sklearn[1.5].utils.validation._num_features
    # message = f"Unable to find the number of features from X of type {type(X)}"
    # # Do not consider an array-like of strings or dicts to be a 2D array
    # if isinstance(X[0], (str, bytes, dict)):
    #     message += f" where the samples are of type {type(X[0]).__qualname__}"
    #     raise TypeError(message)








