# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



def ensure_2D(
    X,
    copy_X:bool=True
):

    """
    Ensure that X has 2-dimensional shape, i.e., len(X.shape) == 2.
    If X is a 1D vector, assume the vector is a single feature of
    samples, not a single sample of features. X must have a 'shape'
    attribute. If copy_X is True and X is 1-dimensional, then X must
    have a 'copy' method. This module does not accept python builtin
    iterables like list, set, and tuple.


    Parameters
    ----------
    X:
        {array-like} of shape (n_samples, n_features) or (n_samples,) -
        The data to be put into a 2-dimensional container.
    copy:
        bool - whether to copy X or operate directly on the passed X.


    Return
    ------
    X:
        {array-like} of shape (n_samples, n_features) - The data in
        a 2-dimensional container.


    """

    # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **
    try:
        iter(X)
        if isinstance(X, (str, dict, set, tuple, list)):
            raise Exception
    except:
        raise ValueError(
            f"ensure_2D: 'X' must be an iterable data-bearing container. "
            f"python builtin iterables are not allowed. Got {type(X)}."
        )


    if not hasattr(X, 'shape'):
        raise ValueError(f"ensure_2D: 'X' must have a 'shape' attribute.")

    if not isinstance(copy_X, bool):
        raise TypeError(f"'copy_X' must be boolean.")

    # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **


    _dim = len(X.shape)

    if _dim == 0:
        raise ValueError(
            f"ensure_2D: 'X' is zero dimensional. Cannot convert 0 "
            f"to 2D."
        )
    elif _dim == 1:

        if not hasattr(X, 'copy'):
            raise ValueError(f"'X' must have a 'copy' method.")

        if copy_X:
            _X = X.copy()
        else:
            _X = X

        # dataframes dont have reshape
        try:
            return _X.reshape((-1, 1))
        except:
            try:
                return _X.to_frame()
            except:
                raise ValueError(
                    f"ensure_2D: unable to cast X to 2D"
                )
    elif _dim == 2:
        return X
    elif _dim > 2:
        raise ValueError(
            f"ensure_2D: 'X' must be 2D or less, got {_dim}. Cannot "
            f"convert 3D+ to 2D."
        )

































