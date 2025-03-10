# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from .._type_aliases import (
    XContainer,
    UpperType
)

from ._upper import _val_upper

from .....base._check_dtype import check_dtype



def _validation(
    _X: XContainer,
    _upper: UpperType
) -> None:


    """
    Centralized hub for validation. See the individual modules for more
    details.


    Parameters:
    -----------
    _X:
        XContainer - the data.
    _upper:
        UpperType - if True, covert all text to upper-case; if False,
        convert all text to lower-case; if None, do a no-op.


    Return
    ------
    -
        None


    """


    check_dtype(_X, allowed='str')

    _val_upper(_upper)






