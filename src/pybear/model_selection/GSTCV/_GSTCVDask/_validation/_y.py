# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from dask import compute
import dask.array as da
import dask.dataframe as ddf



def _val_y(_y) -> None:

    """
    Validate y.

    v v v v pizza keep ur finger on this v v v v
    y must be numeric.

    y must be a single label and binary in 0,1.


    Parameters
    ----------
    _y:
        vector-like of shape (n_samples, 1) or (n_samples,) - The target
        for the data. Must be a dask object.


    Return
    ------
    -
        None

    """


    # y ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    try:
        y_shape = compute(*_y.shape)
    except:
        try:
            y_shape = compute(*da.array(_y).shape)
        except:
            raise TypeError(f"'y' must have a 'shape' attribute pizza fix this")

    # pizza make be any 2 uniques
    _err_msg = (
        f"GSTCVDask can only perform thresholding on vector-like binary "
        f"targets with values in [0,1]. \nPass 'y' as a vector of 0's and 1's."
    )

    if len(y_shape) == 1:
        pass
    elif len(y_shape) == 2:
        if y_shape[1] != 1:
            raise ValueError(_err_msg)
    else:
        raise ValueError(_err_msg)

    # pizza make be any 2 uniques
    if isinstance(_y, da.core.Array):
        _unique = set(da.unique(_y).compute())
    elif isinstance(_y, (ddf.DataFrame, ddf.Series)):
        _unique = set(da.unique(_y.to_dask_array(lengths=True)).compute())

    if not _unique.issubset({0, 1}):

        raise ValueError(_err_msg)

    del _unique


