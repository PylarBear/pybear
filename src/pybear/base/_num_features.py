# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#




def num_features(X) -> int:
    """Return the number of features in an array-like X.

    X must have a 'shape' attribute.

    numpy & pandas: X must be 1 or 2 dimensional.
    scipy: X must be 2 dimensional.
    If X is a 1D vector (i.e., len(shape)==1), return 1.

    Parameters
    ----------
    X : array_like of shape (n_samples, n_features) or (n_samples,)
        Object to find the number of features in, that has a 'shape'
        attribute.

    Returns
    -------
    features : int
        Number of features.

    """


    try:
        X.shape
    except:
        raise ValueError(
            f"\nThe passed object does not have a 'shape' attribute. "
            f"\nAll pybear estimators and transformers require data-bearing "
            f"objects to have a 'shape' attribute, like numpy arrays, pandas "
            f"dataframes, and scipy sparse matrices / arrays."
        )


    if hasattr(X, 'toarray') and len(X.shape) != 2:  # is scipy sparse
        # there is inconsistent behavior with scipy array/matrix and 1D.
        # in some cases scipy.csr_array is allowing 1D to be passed and
        # in other cases it is not. scipy.csr_matrix takes a 1D and reports
        # 2D shape. avoid the whole issue, force all scipy to be 2D.
        raise ValueError(
            f"pybear requires all scipy sparse objects be 2 dimensional"
        )


    if len(X.shape) == 1:
        return 1
    elif len(X.shape) == 2:
        return int(X.shape[1])
    else:
        raise ValueError(
            f"The passed object has {len(X.shape)} dimensions. pybear "
            f"requires that all data-bearing objects be 1 or 2 dimensional."
        )



    # keep this for reference
    # from sklearn[1.5].utils.validation._num_features
    # message = f"Unable to find the number of features from X of type {type(X)}"
    # # Do not consider an array-like of strings or dicts to be a 2D array
    # if isinstance(X[0], (str, bytes, dict)):
    #     message += f" where the samples are of type {type(X[0]).__qualname__}"
    #     raise TypeError(message)








