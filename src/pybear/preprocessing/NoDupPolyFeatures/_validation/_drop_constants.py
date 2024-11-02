# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



def _val_drop_constants(
    _drop_constants: bool
) -> None:

    """
    Validate drop_constants; must be bool


    Parameters
    ----------
    _drop_constants:
        bool, default = False -


    Return
    ------
    -
        None


    """


    if not isinstance(_drop_constants, bool):
        raise TypeError(f"'drop_constants' must be bool")





