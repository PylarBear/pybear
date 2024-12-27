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
        bool -

        When comparing two columns for equality:

        If equal_nan is True, assume that a nan value would otherwise be
        the same as the compared non-nan counterpart, or if both compared
        values are nan, consider them as equal (contrary to the default
        numpy handling of nan, where numpy.nan != numpy.nan).
        If equal_nan is False and either one or both of the values in
        the compared pair of values is/are nan, consider the pair to be
        not equivalent, thus making the column pair not equal. This is
        in line with the normal numpy handling of nan values.

        When assessing if a column is constant:

        If equal_nan is True, assume any nan values are the mean of all
        non-nan values in the respective column. If equal_nan is False,
        any nan-values could never take the value of the mean of the
        non-nan values in the column, making the column not constant.


    Return
    ------
    -
        None


    """


    if not isinstance(_equal_nan, bool):
        raise TypeError(f"'equal_nan' must be bool")









