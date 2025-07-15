# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Literal
from typing_extensions import Union
from ..._type_aliases import ClassifierProtocol

import numbers

from ._sk_estimator import _val_sk_estimator
from ._pre_dispatch import _val_pre_dispatch



def _validation(
    _estimator: ClassifierProtocol,
    _pre_dispatch: Union[Literal['all'], str, numbers.Integral]
) -> None:
    """Centralized hub for sklearn GSTCV validation.

    See the submodules for more information.

    Parameters
    ----------
    _estimator : ClassifierProtocol
        The estimator to be validated.
    _pre_dispatch : Union[Literal['all'], str, numbers.Integral]
        The number of batches (of tasks) to be pre-dispatched. See the
        joblib.Parallel docs for more information.

    Returns
    -------
    None

    """


    _val_sk_estimator(_estimator)

    _val_pre_dispatch(_pre_dispatch)






