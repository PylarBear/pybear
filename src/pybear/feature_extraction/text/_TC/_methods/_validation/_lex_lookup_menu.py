# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing_extensions import Union



def _lex_lookup_menu_validation(
    _possible: str,
    _allowed: Union[str, None],
    _disallowed: Union[str, None]
) -> None:

    """
    Validate lex_lookup_menu parameters. 'allowed' and 'disallowed'
    cannot simultaneously be strings. If both are simultaneously None,
    then all keys are accessible. All inputs are case-sensitive.


    Parameters
    ----------
    _possible:
        str - the keys of the lex_lookup_menu dict.
    _allowed:
        Union[str, None] - keys of the menu that are allowed to be
        accessed.
    _disallowed:
        Union[str, None] - keys of the menu that are not allowed to be
        accessed.


    Return
    ------
    -
        None

    """


    if not isinstance(_possible, str):
        raise TypeError(f"'_possible' must be str")
    if len(_possible) == 0:
        raise ValueError(f"'_possible' cannot be an empty string")


    if _allowed is not None and _disallowed is not None:
        raise ValueError(
            f"cannot enter both 'allowed' and 'disallowed', only one or "
            f"the other or neither."
        )
    elif _allowed is None and _disallowed is None:
        pass
    elif _allowed is not None:

        # VALIDATE ENTRY FOR allowed kwarg #########################
        _base_err = f"lex_lookup_menu() kwarg 'allowed' "
        if not isinstance(_allowed, str):
            raise TypeError(_base_err + f"must be str")
        if len(_allowed) == 0:
            raise ValueError(_base_err + f"cannot be an empty string")

        for _char in _allowed:
            if _char not in _possible:
                raise ValueError(
                    _base_err + f"has invalid key '{_char}', can only "
                    f"contain any of '{_possible}'"
                )
        del _base_err
        # END VALIDATE ENTRY FOR allowed kwarg #####################

    elif _disallowed is not None:
        # VALIDATE ENTRY FOR disallowed kwarg ######################
        _base_err = f"lex_lookup_menu() kwarg 'disallowed' "
        if not isinstance(_disallowed, str):
            raise TypeError(_base_err + f"must be str")
        if len(_disallowed) == 0:
            raise ValueError(_base_err + f"cannot be an empty string")

        for _char in _disallowed:
            if _char not in _possible:
                raise ValueError(
                    _base_err + f"has invalid key '{_char}', can only "
                    f"contain any of '{_possible}'"
                )
        del _base_err
        # END VALIDATE ENTRY FOR disallowed kwarg ##################






