# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing_extensions import Union

import numbers



def _val_sep_flags(
    _sep_flags: Union[numbers.Integral, None]
) -> None:

    """
    Validate 'sep_flags'. Must be an integer or None.


    Parameters
    ----------
    _sep_flags:
        Union[numbers.Integral, None] - the flags for 'sep'.


    Returns
    -------
    -
        None


    """


    err_msg = f"'sep_flags' must be an integer or None."

    if not isinstance(_sep_flags, (numbers.Integral, type(None))):
        raise TypeError(err_msg)

    if isinstance(_sep_flags, bool):
        raise TypeError(err_msg)







