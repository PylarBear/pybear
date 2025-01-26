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



