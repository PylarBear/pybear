# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


def _val_conflict(_conflict: str) -> None:


    # must be 'raise' or 'ignore'

    err_msg = f"'conflict' must be literal 'raise' or 'ignore'"

    if not isinstance(_conflict, str):
        raise TypeError(err_msg)

    if _conflict.lower() not in ['raise', 'ignore']:
        raise ValueError(err_msg)





