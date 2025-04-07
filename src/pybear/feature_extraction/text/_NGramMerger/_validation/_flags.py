# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing_extensions import Union

import numbers



def _val_flags(
    _flags: Union[numbers.Integral, None]
) -> None:

    """
    Validate 'flags'. Must be an integer or None.


    Parameters
    ----------
    _flags:
        Union[numbers.Integral, None] - the global flags value(s) applied
        to the n-gram search. Must be None or an integer. The values of
        the integers are not validated for legitimacy, any exceptions
        would be raised by re.fullmatch.


    Returns
    -------
    -
        None

    """


    if _flags is None:
        return


    if not isinstance(_flags, numbers.Integral) or isinstance(_flags, bool):
        raise TypeError(
            f"'flags' must be None or an integer. got {type(_flags)}."
        )










