# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



def ensure_2D(OBJECT):

    """
    Ensure that OBJECT has 2 dimensional shape, i.e.,
    len(OBJECT.shape) == 2.
    If OBJECT is a 1D vector, assume the vector is a single feature of
    samples, not a single sample of features.


    Parameters
    ----------
    OBJECT:
        {array-like} of shape (n_samples, n_features) or (n_samples,) -
        The data.


    Return
    ------
    OBJECT:
        {array-like} of shape (n_samples, n_features) - The data in
        a 2 dimensional array.


    """


    try:
        iter(OBJECT)
        if isinstance(OBJECT, (str, dict)):
            raise Exception
    except:
        raise ValueError(
            f"ensure_2D: 'OBJECT' must be an iterable data-bearing "
            f"object. Got {type(OBJECT)}."
        )


    if not hasattr(OBJECT, 'shape'):
        raise ValueError(f"ensure_2D: 'OBJECT' must have a 'shape' attribute.")


    _dim = len(OBJECT.shape)
    if _dim == 0:
        raise ValueError(
            f"ensure_2D: 'OBJECT' is zero dimensional. Cannot convert 0 "
            f"to 2D."
        )
    elif _dim == 1:
        # data frames dont have reshape
        try:
            return OBJECT.reshape((-1, 1))
        except:
            try:
                return OBJECT.to_frame()
            except:
                raise ValueError(
                    f"ensure_2D: unable to cast OBJECT to 2D"
                )
    elif _dim == 2:
        return OBJECT
    elif _dim > 2:
        raise ValueError(
            f"ensure_2D: 'OBJECT' must be 2D or less, got {_dim}. Cannot "
            f"convert 3D+ to 2D."
        )


    return OBJECT
































