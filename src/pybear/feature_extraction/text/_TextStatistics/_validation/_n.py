# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import numbers



def _val_n(
    n: numbers.Integral
) -> None:


    err_msg = f"'n' must be an integer >= 1"
    if not isinstance(n, numbers.Integral):
        raise TypeError(err_msg)
    if isinstance(n, bool):
        raise TypeError(err_msg)
    if n < 1:
        raise ValueError(err_msg)
    del err_msg




