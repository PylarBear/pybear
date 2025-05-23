# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#




def _val_equal_nan(
    _equal_nan: bool
) -> None:

    """
    Validate equal_nan; must be bool.


    Parameters
    ----------
    _equal_nan:
        bool - If equal_nan is True, exclude nan-likes from computations
        that discover constant columns. This essentially assumes that
        the nan value would otherwise be equal to the mean of the non-nan
        values in the same column. If equal_nan is False and any value
        in a column is nan, do not assume that the nan value is equal to
        the mean of the non-nan values in the same column, thus making
        the column non-constant. This is in line with the normal numpy
        handling of nan values.


    Return
    ------
    -
        None


    """


    if not isinstance(_equal_nan, bool):
        raise TypeError(f"'equal_nan' must be bool")





