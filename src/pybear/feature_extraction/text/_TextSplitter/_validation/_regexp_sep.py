# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Sequence
from .._type_aliases import RegExpSepType

import re



def _val_regexp_sep(
    _regexp_sep: RegExpSepType,
    _X: Sequence[str]
) -> None:

    """
    Validate the regexp parameter for re.strip(). Must be None, a single
    regexp string, a re.Pattern object, or a list of regexp strings,
    re.Pattern instances, or Falses. If passed as a list, the number of
    entries must equal the number of strings in X.


    Parameters
    ----------
    _regexp_sep:
        RegExpSepType - the regexp string(s) or the re.Pattern object(s)
        to split the strings in X on. If a single expression or pattern,
        that is applied to all strings in X. If passed as a list, the
        number of entries must equal the number of strings in X, and the
        entries are applied to the corresponding string in X. The regular
        expressions and the re.Pattern objects themselves are not
        validated for legitimacy, any exceptions would be raised by
        re.split().


    Return
    ------
    -
        None


    Notes
    -----
    see re.split()

    """


    if _regexp_sep is None:
        return


    err_msg = (
        f"'regexp_sep' must be None, a single regular expression, a "
        f"re.Pattern object, or a list of regexp strings, re.Pattern "
        f"objects and Falses. If passed as a list, the number of entries "
        f"must equal the number of strings in X."
    )

    try:
        if isinstance(_regexp_sep, re.Pattern):
            raise UnicodeError
        iter(_regexp_sep)
        if isinstance(_regexp_sep, dict):
            raise Exception
        if isinstance(_regexp_sep, str):
            raise UnicodeError
        raise TimeoutError
    except UnicodeError:
        # if is a single string or re.Pattern
        pass
    except TimeoutError:
        # if is a sequence of somethings
        if len(_regexp_sep) != len(_X):
            raise ValueError(err_msg)
        for _ in _regexp_sep:
            if not isinstance(_, (str, re.Pattern, bool)):
                raise TypeError(err_msg)
            if _ is True:
                raise TypeError(err_msg)
    except Exception as e:
        raise TypeError(err_msg)



    del err_msg









