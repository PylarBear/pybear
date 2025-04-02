# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



def _val_any_bool(
    _any_bool: bool,
    _name: str = 'unnamed_boolean'
) -> None:

    """
    Validate '_any_bool'. Must be boolean.


    Parameters
    ----------
    _any_bool:
        bool - something that can only be boolean.


    Returns
    -------
    -
        None


    """


    if not isinstance(_any_bool, bool):
        raise TypeError(f"'{_name}' must be boolean.")






