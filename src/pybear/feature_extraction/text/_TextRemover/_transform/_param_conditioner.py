# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Literal
from typing_extensions import Union
from .._type_aliases import (
    RegExpType,
    RegExpFlagsType
)

import numbers
import re



def _param_conditioner(
    _regexp_remove: Union[RegExpType, list[Union[RegExpType, Literal[False]]]],
    _regexp_flags: RegExpFlagsType,
    _n_rows: numbers.Integral
): #-> tuple[pizza, pizza]:

    """
    
    
    Returns
    -------

    """


    # map the given regexp params to list, if not list already
    # convert re.fullmatch params to lists -- -- -- -- -- -- -- -- --

    # _regexp_remove must be str, re.Pattern, list[Union[str, re.Pattern, False]]
    # _regexp_remove cannot be None. The code that allows entry into this
    # module explicitly says "if _regexp_remove is not None:".

    if isinstance(_regexp_remove, (str, re.Pattern)):
        _remove = [_regexp_remove for _ in range(_n_rows)]
    elif isinstance(_regexp_remove, set):
        _remove = ['|'.join(map(re.escape, _regexp_remove)) for _ in range(_n_rows)]
    elif _regexp_remove is False:
        # this is a fail-safe. rr could only be False if X is 2D, rr is
        # a list, and we have sent one of the rows of X and its rr value
        # into here. but in main TR.transform False is explicitly skipped
        _remove = [_regexp_remove for _ in range(_n_rows)]
    elif isinstance(_regexp_remove, list):
        _remove = _regexp_remove
    else:
        raise Exception

    # but _regexp_flags definitely can be None (uses re.fullmatch default flags)
    if isinstance(_regexp_flags, (type(None), numbers.Integral)):
        _flags = [_regexp_flags for _ in range(_n_rows)]
    elif _regexp_flags is False:
        # this is a fail-safe. rf could only be False if X is 2D, rf is
        # a list, and we have sent one of the rows of X and its rf value
        # into here. but in main TR.transform False is explicitly skipped
        _flags = [_regexp_flags for _ in range(_n_rows)]
    elif isinstance(_regexp_flags, list):
        _flags = _regexp_flags
    else:
        raise Exception

    # END convert re.fullmatch params to lists -- -- -- -- -- -- -- --


    return _remove, _flags


