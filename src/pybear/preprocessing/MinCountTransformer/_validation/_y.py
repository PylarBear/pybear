# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



# pizza this is a fluid situation
# need to finalize containers, and in validate_data min rows



from .._type_aliases import YContainer

import numpy as np
import pandas as pd



def _val_y(
    y: YContainer
) -> None:


    if not isinstance(
        y,
        (
            type(None),
            np.ndarray,
            pd.core.frame.DataFrame,
            pd.core.series.Series
        )
    ):
        raise TypeError(
            f'invalid data container for y, {type(y)}. pizza placeholder, '
            f'finalize allowed containers'
        )


    # pizza
    # validate_data() reshapes y to 2D always (for now)