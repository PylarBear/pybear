# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing_extensions import Union



def _menu_validation(
    _possible: str,
    _allowed: Union[str, None],
    _disallowed: Union[str, None]
) -> None:

    """
    Validate parameters for the 'menu' method. Cannot pass both 'allowed'
    and 'disallowed' simultaneously, any entries must be string with
    characters in the keys of MENU_DICT. If neither is passed, all menu
    options are available. All inputs are all case-sensitive.


    Parameters
    ----------
    _possible:
        str - all keys of MENU_DICT
    _allowed:
        Union[str, None] - the keys of MENU_DICT that are allowed to be
        accessed.
    _disallowed:
        Union[str, None] - the keys of MENU_DICT that are not allowed to
        be accessed.


    Return
    ------
    -
        None

    """


    # pizza verify the alloweds
    # alloweds = "ABCDEFIJLNOPQRSTUWXZ"

    _base_err = f"menu() kwarg '_possible' "
    if not isinstance(_possible, str):
        raise TypeError(_base_err + f"must be a string")
    if not len(_possible):
        raise ValueError(_base_err + f"cannot be an empty string")
    del _base_err

    if _allowed is not None and _disallowed is not None:
        raise ValueError(
            f"Cannot enter both 'allowed' and 'disallowed', must be one "
            f"or the other or neither."
        )
    elif _allowed is not None:
        _base_err = f"menu() kwarg 'allowed' "
        if not isinstance(_allowed, str):
            raise TypeError(_base_err + f"must be str")
        if len(_allowed) == 0:
            raise ValueError(_base_err + f"cannot be an empty string")

        for _char in _allowed:
            if _char not in _possible:
                raise ValueError(
                    _base_err + f"has invalid key '{_char}', must be any "
                    f"of {_possible}."
                )
        del _base_err
    elif _disallowed is not None:
        _base_err = f"menu() kwarg 'disallowed' "
        if not isinstance(_disallowed, str):
            raise TypeError(_base_err + f"must be str")
        if len(_disallowed) == 0:
            raise ValueError(_base_err + f"cannot be an empty string")

        for _char in _disallowed:
            if _char not in _possible:
                raise ValueError(
                    _base_err + f"has invalid key '{_char}', must be any "
                    f"of {_possible}."
                )
        del _base_err



















