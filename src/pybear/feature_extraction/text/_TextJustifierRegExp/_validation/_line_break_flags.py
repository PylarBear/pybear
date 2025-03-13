# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing_extensions import Union

import numbers



def _val_line_break_flags(
    _line_break_flags: Union[numbers.Integral, None]
) -> None:

    """
    Validate 'line_break_flags'. Must be an integer or None.


    Parameters
    ----------
    _line_break_flags:
        Union[numbers.Integral, None] - the flags for 'line_break'.


    Returns
    -------
    -
        None


    """


    err_msg = f"'line_break_flags' must be an integer or None."

    if not isinstance(_line_break_flags, (numbers.Integral, type(None))):
        raise TypeError(err_msg)

    if isinstance(_line_break_flags, bool):
        raise TypeError(err_msg)







