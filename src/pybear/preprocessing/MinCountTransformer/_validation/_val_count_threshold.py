# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


def _val_count_threshold(_count_threshold: int) -> int:

    """
    Validate count_threshold is non-bool integer >= 2.

    """

    err_msg = f"count_threshold must be an integer >= 2"

    if isinstance(_count_threshold, bool):
        raise TypeError(err_msg)

    try:
        float(_count_threshold)
        if not int(_count_threshold) == _count_threshold:
            raise Exception
        _count_threshold = int(_count_threshold)
    except:
        raise TypeError(err_msg)

    if not _count_threshold >= 2:
        raise ValueError(err_msg)
    del err_msg

    return _count_threshold









