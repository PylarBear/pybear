# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing_extensions import Union
from ..._type_aliases import (
    CaseSensitiveType,
    FlagsType
)

import re



def _flag_maker(
    _remove: list[Union[list[None], list[re.Pattern]]],
    _case_sensitive: CaseSensitiveType,
    _flags: FlagsType
) -> list[Union[list[None], list[re.Pattern]]]:

    """
    Use flags inferred from _case_sensitive and any user-passed flags
    to put flags in the re.compile objects in _remove. All string literals
    that were in _remove must already be converted to re.compile. _remove
    can only contain [None]s and list[re.Pattern]s.


    Parameters
    ----------
    _remove:
        RemoveType - the string removal criteria converted entirely so
        that row-wise, _remove is comprised of list[re.Pattern]s and
        [None]s.
    _case_sensitive:
        CaseSensitiveType - the case-sensitive strategy as passed by the
        user.
    _flags:
        FlagsType - the flags for regex searches as passed by the user.


    Returns
    -------
    -
        list[Union[list[None], list[re.Pattern]]] - the _remove object
        with the appropriate flags now in every re.compile object.


    """


    if _remove is None:
        raise TypeError(
            f"'_remove' is None, should have been handled elsewhere"
        )

    # if _case_sensitive and/or _flags are lists, the length was validated
    # against the data previously.

    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

    # get the setting for case_sensitive for the row in _X. if cs is list,
    # a value could be None or bool: set None to True

    for _idx, _row in enumerate(_remove):

        if isinstance(_case_sensitive, bool):
            _cs_flags = re.I if _case_sensitive is False else 0
        elif isinstance(_case_sensitive, list):
            # if cs in list is True or None, keep case_sensitive (no flag)
            _cs_flags = re.I if _case_sensitive[_idx] is False else 0
        else:
            raise Exception

        if _flags is None:
            _og_flags = 0
        elif isinstance(_flags, type(re.X)):
            _og_flags = _flags
        elif isinstance(_flags, list):
            _og_flags = _flags[_idx] or 0
        else:
            raise Exception


        _new_flags = _og_flags | _cs_flags

        # go thru the list in every row of _remove. put the flags in.
        for _inner_idx, _inner_thing in enumerate(_row):
            if _inner_thing is None:
                continue
            elif isinstance(_inner_thing, re.Pattern):
                _remove[_idx][_inner_idx] = re.compile(
                    _inner_thing.pattern,
                    _inner_thing.flags | _new_flags
                )
            else:
                raise Exception(f"algorithm failure.")


    return _remove





