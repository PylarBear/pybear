# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from .._type_aliases import YContainer

import warnings
import numpy as np
import pandas as pd



def _val_y(
    _y: YContainer
) -> None:

    """
    Validate the target for the data is a valid data container. Numpy
    ndarrays, pandas dataframes, and pandas series are allowed. This
    validation is performed for :meth: `partial_fit` and :meth: `fit`
    even though y is ignored. This validation is also performed
    for :meth: `transform` and is necessary because y may be passed to
    transform and be reduced along the sample axis.


    Parameters
    ----------
    _y:
        Union[numpy.ndarray, pandas.DataFrame, pandas.Series, None] of
        shape (n_samples, n_features) or (n_samples,). The target for
        the data. Ignored in :meth: `partial_fit` and :meth: `fit`,
        optional for :meth: `transform`.


    Return
    ------
    -
        None

    """


    if not isinstance(
        _y,
        (
            type(None),
            np.ndarray,
            pd.core.frame.DataFrame,
            pd.core.series.Series
        )
    ):
        raise TypeError(f'invalid data container for y, {type(_y)}.')

    if isinstance(_y, np.rec.recarray):
        raise TypeError(
            f"MCT does not accept numpy recarrays. "
            f"\npass your data as a standard numpy array."
        )

    if np.ma.isMaskedArray(_y):
        warnings.warn(
            f"MCT does not block numpy masked arrays but they are not tested. "
            f"\nuse them at your own risk."
        )



