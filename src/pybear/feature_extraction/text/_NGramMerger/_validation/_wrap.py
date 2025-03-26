# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



def _val_wrap(_wrap: bool) -> None:

    """
    Validate wrap. Must be boolean.


    Parameters
    ----------
    _wrap:
        bool - whether to look for pattern matches across the end of the
        current line and beginning of the next line.


    Returns
    -------
    -
        None


    """


    if not isinstance(_wrap, bool):
        raise TypeError(f"'wrap' must be boolean")





