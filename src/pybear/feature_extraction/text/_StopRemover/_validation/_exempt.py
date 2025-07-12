# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Sequence
from typing_extensions import Union

from .....base._check_1D_str_sequence import check_1D_str_sequence



def _val_exempt(_exempt: Union[Sequence[str], None]) -> None:
    """Validate `exempt`.

    Must be sequence[str] or None.

    Parameters
    ----------
    _exempt : Union[Sequence[str], None]
        Stop words that are exempt from being removed.

    Returns
    -------
    None

    """


    if _exempt is None:
        return


    check_1D_str_sequence(_exempt, require_all_finite=False)






