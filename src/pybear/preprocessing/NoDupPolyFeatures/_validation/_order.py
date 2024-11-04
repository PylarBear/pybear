# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from typing import Literal




def _val_order(_order: Literal['C', 'F']) -> None:

    """

    Validate 'order' is string literal in ['C', 'F']


    Parameters
    ----------
    _order:
        Literal['C', 'F'] -

    Return
    ------
    -
        None

    """

    err_msg = f"'order' must be literal 'C' or 'F'"

    if not isinstance(_order, str):
        raise ValueError(err_msg)

    if not _order.lower() in 'cf':
        raise ValueError(err_msg)




