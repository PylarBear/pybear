# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#




def _val_equal_nan(
    _equal_nan: bool
) -> None:

    """
    Validate equal_nan; must be bool


    Parameters
    ----------
    _equal_nan:
        bool, default = False - When comparing pairs of columns row by
        row:
        If equal_nan is True, exclude from comparison any rows where one
        or both of the values is/are nan. If one value is nan, this
        essentially assumes that the nan value would otherwise be the
        same as its non-nan counterpart. When both are nan, this
        considers the nans as equal (contrary to the default numpy
        handling of nan, where np.nan != np.nan) and will not in and of
        itself cause a pair of columns to be marked as unequal.
        If equal_nan is False and either one or both of the values in
        the compared pair of values is/are nan, consider the pair to be
        not equivalent, thus making the column pair not equal. This is
        in line with the normal numpy handling of nan values.


    Return
    ------
    -
        None


    """


    if not isinstance(_equal_nan, bool):
        raise TypeError(f"'equal_nan' must be bool")









