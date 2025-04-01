# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing_extensions import Union
from ..._type_aliases import (
    RemoveType,
    CaseSensitiveType,
    FlagsType
)

import numbers
import re

from ._compile_maker import _compile_maker
from ._flag_maker import _flag_maker



def _param_conditioner(
    _remove: RemoveType,
    _case_sensitive: CaseSensitiveType,
    _flags: FlagsType,
    _n_rows: numbers.Integral
) -> Union[
        None, re.Pattern[str], tuple[re.Pattern, ...],
        list[Union[None, re.Pattern[str], tuple[re.Pattern, ...]]]
    ]:

    """
    Use the parameters to convert all literal strings to re.compile and
    apply the flags implied by _case_sensitive and _flags.


    Parameters
    ----------
    _remove:
        RemoveType - the string removal criteria as passed by the user.
    _case_sensitive:
        CaseSensitiveType - the case-sensitive strategy as passed by the
        user.
    _flags:
        FlagsType - the flags for regex searches as passed by the user.
    _n_rows:
        numbers.Integral - the number of rows in the data. if _remove,
        _case_sensitive and/or _flags were passed as a list, the length
        of them was already validated against this number.

    
    Returns
    -------
    -
        _remove: Union[None, re.Pattern[str], tuple[re.Pattern, ...],
            list[Union[None, re.Pattern[str], tuple[re.Pattern, ...]]] -
            the removal criteria for the data. Could be a single None,
            as single re.Pattern, as single tuple of re.Patterns, or
            a list comprised of any of those things.
    ]


    """

    # dont need validation. these parameters come directly from the
    # instance parameters which are validated in _validation.

    # map the given params to re.Pattern objects if _remove is not None
    # only return the output as a list if absolutely necessary

    # v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^
    # '_remove' must be None, str, re.compile, tuple or list
    # '_case_sensitive' must be bool, list[None or bool]
    # '_flags' must be None, int, or list[None or int]
    # v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^

    if _remove is None:
        # from validation we know that '_flags' must be None and
        # '_case_sensitive' cannot be list. they dont matter. there is
        # nothing to remove, a no-op.
        return None

    # v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^
    # '_remove' must be str, re.compile, tuple[str, compile] or list
    # '_case_sensitive' must be bool, list[None or bool]
    # '_flags' must be None, int, or list[None or int]
    # v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^

    # convert '_remove' to list, if not already, holding only None/compile.
    # in the process:
    #   convert any associated str into flagless re.compile (re.escape!)
    #   ensure any tuples dont have duplicates (use set on re.compile, it works)
    #   make everything inside the outer list be in a list (so None becomes
    #   [None], compile becomes [compile] and tuple becomes list.

    _remove = _compile_maker(_remove, _n_rows)

    # v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^
    # '_remove' must be list[Union[[None], [re.compile[str]]]]
    # '_case_sensitive' must be bool, list[None or bool]
    # '_flags' must be None, int, or list[None or int]
    # v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^

    # set the flags for the re.compile objects
    # '_flags' always trumps '_case_sensitive'. if user passed re.I in any way
    # (global or to individual rows) that trumps global '_case_sensitive' == True.
    # if '_case_sensitive' == False, everything gets re.I.

    _remove = _flag_maker(_remove, _case_sensitive, _flags)

    # pull the inner objects out of their lists. if was tuple, turn
    # that back to a tuple
    for _idx, _row in enumerate(_remove):

        if len(_remove[_idx]) == 0:
            raise Exception('algorithm failure')
        elif len(_remove[_idx]) == 1:
            _remove[_idx] = _remove[_idx][0]
        else:
            _remove[_idx] = tuple(_remove[_idx])

    # v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^
    # '_remove' must be list[Union[None, re.compile[str], tuple[compile, ...]]]
    # '_case_sensitive' doesnt matter anymore
    # '_flags' doesnt matter anymore
    # v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^


    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    # look to see if _remove is unnecessarily iterable, meaning
    # all the values in _remove are identical

    if len(_remove) and all(map(lambda x: x == _remove[0], _remove)):
        _remove = _remove[0]


    return _remove








