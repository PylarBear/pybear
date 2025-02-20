# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing_extensions import Union



def _remove_characters_validation(
    _allowed_chars: Union[str, None],
    _disallowed_chars: Union[str, None]
) -> None:

    """
    The characters to be kept or the characters to be removed from the
    data. Case-sensitive. '_allowed_chars' and '_disallowed_chars' cannot
    be simultaneously None or simultaneously strings.


    Parameters
    ----------
    _allowed_chars:
        Union[str, None] - the characters that are allowed to stay in
        the data.
    _disallowed_chars:
        Union[str, None] the characters that are to be removed from the
        data.


    Return
    ------
    -
        None


    """

    if _allowed_chars is None and _disallowed_chars is None:
        raise ValueError(
            f"Must specify at least one of 'allowed_chars' or 'disallowed_chars'"
        )
    elif _allowed_chars is not None and _disallowed_chars is not None:
        raise ValueError(
            f"Cannot enter both 'allowed_chars' and 'disallowed_chars'. "
            f"Only one or the other.'"
        )
    elif _allowed_chars is not None:
        if not isinstance(_allowed_chars, str):
            raise TypeError(f"'allowed_chars' must be str")
        if len(_allowed_chars) == 0:
            raise ValueError(f"'allowed_chars' cannot be an empty string")
    elif _disallowed_chars is not None:
        if not isinstance(_disallowed_chars, str):
            raise TypeError(f"'disallowed_chars' must be str")
        if len(_disallowed_chars) == 0:
            raise ValueError(f"'disallowed_chars' cannot be an empty string")






