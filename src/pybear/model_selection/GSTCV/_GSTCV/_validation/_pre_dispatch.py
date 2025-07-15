# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#




from typing import (
    Literal,
    Union
)

import numbers



def _val_pre_dispatch(
    _pre_dispatch:Union[Literal['all'], str, numbers.Integral]
) -> None:
    """Validate `_pre_dispatch`.

    This file is a placeholder. There is no validation here, any errors
    would be raised by joblib.Parallel().

    Parameters
    ----------
    _pre_dispatch : Union[Literal['all'], str, numbers.Integral]
        The number of batches (of tasks) to be pre-dispatched. See the
        joblib.Parallel docs for more information.

    Returns
    -------
    None

    """


    return









