# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from typing import Union

from ....text import alphanumeric_str as ans



def _lex_lookup_menu(
        LEX_LOOK_DICT: dict[str, str],
        allowed: Union[str, None]=None,
        disallowed: Union[str, None]=None
    ) -> tuple[str, str]:


    # ALLOWED ARE 'adelks' ---- pizza what this mean?


    """
    Dynamic function for returning variable menu prompts and allowed
    commands.

    Only one of 'allowed' or 'disallowed' can be passed. Validate entry
    against keys used in LEX_LOOK_DICT.
    If neither 'allowed' nor 'disallowed' is passed, allow any key from
    LEX_LOOK_DICT.

    Parameters
    ----------
    LEX_LOOK_DICT:
        dict[str, str] - dictionary of menu options and prompts for
        lex_lookup
    allowed:
        Union[str, None] - optional, default = None, string of allowed
        keys from LEX_LOOK_DICT. If neither allowed or disallowed is
        provided, all the keys in LEX_LOOK_DICT are allowed.
    disallowed:
        Union[str, None] - optional, default = None, string of disallowed
        keys in LEX_LOOK_DICT; the conjugate set of keys are allowed.

    Return
    ------
    tuple[WIP_DISPLAY:str, allowed:str]
        WIP_DISPLAY: str - prompts to display for allowed menu options
        allowed: str - allowed keys from LEX_LOOK_DICT


    """

    lex_look_allowed = "".join(list(LEX_LOOK_DICT.keys())).lower()

    if allowed is not None and disallowed is not None:
        raise ValueError(f"cannot enter both 'allowed' and 'disallowed', "
                         f"only one or the other or neither")

    elif allowed is None and disallowed is None:
        allowed = lex_look_allowed

    # VALIDATE ENTRY FOR allowed #######################################
    # if allowed is not None, which must be the case at this point
    if not isinstance(allowed, (str, type(None))):
        raise TypeError(f"'allowed' must be a single string or None")

    if isinstance(allowed, str):
        for _ in allowed:

            if _.upper() not in ans.alphabet_str_upper():
                raise ValueError(f"'allowed' can only contain alpha characters")

            if _.upper() not in lex_look_allowed.upper():
                raise ValueError(
                    f"'allowed' can only contain chars from '{lex_look_allowed}'")

    # END VALIDATE ENTRY FOR allowed ###################################

    if disallowed is not None:

        # this is only done to generate a value for allowed

        # VALIDATE ENTRY FOR disallowed ################################

        if not isinstance(disallowed, str):
            raise TypeError(f"'disallowed' must be a single string")

        for _ in disallowed:

            if _.upper() not in ans.alphabet_str_upper():
                raise ValueError(f"'disallowed' can only contain alpha characters")

            if _.upper() not in lex_look_allowed.upper():
                raise ValueError(
                    f"'disallowed' can only contain chars from {lex_look_allowed}"
                )
        # END VALIDATE ENTRY FOR disallowed ############################

        allowed = ''
        for _ in lex_look_allowed.lower():
            if _ not in disallowed:
                allowed += _

    WIP_DISPLAY = []
    for k, v in LEX_LOOK_DICT.items():
        if k.lower() in allowed.lower():
            WIP_DISPLAY.append(f'{v.upper()}({k.lower()})')

    WIP_DISPLAY = ", ".join(WIP_DISPLAY)

    return WIP_DISPLAY, allowed






























