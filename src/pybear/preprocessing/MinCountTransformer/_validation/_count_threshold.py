# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


def _val_count_threshold(
    _count_threshold: int
) -> None:

    """
    Validate count_threshold is non-bool integer >= 2.


    Parameters
    ----------
    _count_threshold:
        int: the minimum frequency a value must have within a column in
        order to not be removed.


    Return
    ------
    -
        None


    """


    try:
        float(_count_threshold)
        if isinstance(_count_threshold, bool):
            raise Exception
        if not int(_count_threshold) == _count_threshold:
            raise Exception
        _count_threshold = int(_count_threshold)
    except:
        raise TypeError(f"count_threshold must be an integer >= 2")

    if not _count_threshold >= 2:
        raise ValueError(f"count_threshold must be an integer >= 2")










