# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#




import numbers



def _split_helper(
    _string: str,
    _idx: numbers.Integral
) -> tuple[str, str]:

    """
    A helper function for managing splits for TextJustifier wraps and
    line breaks. This slices to just after the index, whereas normal
    python slicing is to slice to just before the index.


    Parameters
    ----------
    _string:
        str - the string to be split
    _idx:
        numbers.Integral - the index location to split at, not removing
        the split character string, but keeping it on the left-hand side
        of the split.


    Return
    ------
    -
        tuple[str, str] - the left-hand and right-hand sides of the
        split. If the split is at the far right of '_string' (or
        even beyond), then the right-hand value is an empty string.

    """

    if _idx >= len(_string) - 1:
        return _string, ''
    else:
        return _string[:_idx + 1], _string[_idx + 1:]






