# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



def _val_max_recursions(_max_recursions: int) -> int:

    """
    _max_recursions must be an integer >= 1

    """


    err_msg = f"max_recursions must be an integer >= 1"

    try:
        float(_max_recursions)
        if isinstance(_max_recursions, bool):
            raise Exception
        if int(_max_recursions) != _max_recursions:
            raise Exception
        _max_recursions = int(_max_recursions)
    except:
        raise TypeError(err_msg)

    if not _max_recursions in range(1, 100):
        raise ValueError(err_msg)

    del err_msg


    return _max_recursions





















