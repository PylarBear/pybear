# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from .._type_aliases import (
    RegExpRemoveType,
    XContainer
)

import re



def _val_regexp_remove(
    _regexp_remove: RegExpRemoveType,
    _X: XContainer
) -> None:

    """
    Validate regexp_remove. Must be None, a regexp string, a re.Pattern
    object, or a list of regexp strings, re.Pattern objects and/or Falses.
    The patterns are not validated here, any exception would be raised
    by re.fullmatch. If passed as a list, the number of entries must
    equal the number of strings in X.


    Parameters
    ----------
    _regexp_remove:
        RegExpSepType - the regexp string(s) or the re.Pattern object(s)
        used to match patterns for removal from the data. If a single
        expression or pattern, that is applied to all strings in X. If
        passed as a list, the number of entries must equal the number of
        strings in X, and the entries are applied to the corresponding
        string in X. The regular expressions and the re.Pattern objects
        themselves are not validated for legitimacy, any exceptions
        would be raised by re.fullmatch().


    Return
    ------
    -
        None


    Notes
    -----
    see re.fullmatch()


    """


    if _regexp_remove is None:
        return


    err_msg = (
        f"'regexp_remove' must be None, a single regular expression, a "
        f"re.Pattern object, or a list of regexp strings, re.Pattern "
        f"objects and Falses. If passed as a list, the number of entries "
        f"must equal the number of strings in X."
    )

    try:
        if isinstance(_regexp_remove, (str, re.Pattern)):
            raise UnicodeError
        if isinstance(_regexp_remove, list):
            raise TimeoutError
        raise Exception
    except UnicodeError:
        # if is a single string or re.Pattern
        pass
    except TimeoutError:
        # if is a sequence of somethings
        if len(_regexp_remove) != len(_X):
            raise ValueError(err_msg)
        _types = (str, re.Pattern, bool)
        if not all(map(isinstance, _regexp_remove, (_types for _ in _regexp_remove))):
            raise TypeError(err_msg)
        if any(map(lambda x: x is True, _regexp_remove)):
            raise TypeError(err_msg)
        del _types
    except Exception as e:
        raise TypeError(err_msg)



    del err_msg










