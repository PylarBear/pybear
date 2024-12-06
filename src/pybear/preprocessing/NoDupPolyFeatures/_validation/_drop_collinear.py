# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



def _val_drop_collinear(
    _drop_collinear: bool
) -> None:

    """
    Validate drop_collinear; must be bool


    Parameters
    ----------
    _drop_colinear:
        bool - whether to drop collinear columns from the polynomial feature
        expansion. That is, whether to omit columns that have been multiplied
        by a constant column.


    Return
    ------
    -
        None


    """


    if not isinstance(_drop_collinear, bool):
        raise TypeError(f"'_drop_collinear' must be bool")





