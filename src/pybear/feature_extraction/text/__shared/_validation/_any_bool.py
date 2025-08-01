# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Optional



def _val_any_bool(
    _bool:bool,
    _name:Optional[str] = 'unnamed boolean',
    _can_be_None:Optional[bool] = False
) -> None:
    """Validate '_bool'. Must be boolean.

    Parameters
    ----------
    _bool : bool
        Something that can only be boolean.
    _name : Optional[str], default='unnamed boolean'
        The name of the parameter being validated as boolean, or None if
        allowed.
    _can_be_None : Optional[bool], default=False
        Warnings the boolean value is allowed to be passed as None.

    Returns
    -------
    None

    """


    # validation --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    if not isinstance(_name, str):
        raise TypeError(f"'_name' must be a string. got {type(_name)}.")

    if not isinstance(_can_be_None, bool):
        raise TypeError(f"'_can_be_None' must be bool. got {type(_can_be_None)}.")

    # END validation --- --- --- --- --- --- --- --- --- --- --- --- ---


    if _can_be_None and _bool is None:
        return


    if not isinstance(_bool, bool):
        raise TypeError(f"'{_name}' must be boolean.")






