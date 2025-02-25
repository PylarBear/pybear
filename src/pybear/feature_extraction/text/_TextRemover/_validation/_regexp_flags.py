# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from .._type_aliases import (
    XContainer,
    RegExpFlagsType
)

import numbers



def _val_regexp_flags(
    _regexp_flags: RegExpFlagsType,
    _X: XContainer
) -> None:

    """
    Validate the 'flags' parameter for re.fullmatch(). Must be integer,
    None, or list of integer, None, and literal False. If a list, the
    length must match the length of X.


    Parameters
    ----------
    _regexp_flags:
        RexExpFlagsType - the flags arguments for re.fullmatch(), if
        regular expressions are being used. Must be None, an integer,
        or a list of Nones, integers, or Falses. When passed as a list,
        the length must match the number of strings in the data. The
        values of the integers are not validated for legitimacy, any
        exceptions would be raised by re.fullmatch().
    _X:
        XContainer - the data.


    Return
    ------
    -
        None


    Notes
    -----
    re.fullmatch()

    """


    if _regexp_flags is None:
        return


    err_msg = (
        f"'regexp_flags' must be None, a single integer, or a list that "
        f"contains any combination of Nones, integers, or literal Falses, "
        f"whose length matches the length of X."
    )

    try:
        if isinstance(_regexp_flags, numbers.Integral):
            raise UnicodeError
        if isinstance(_regexp_flags, list):
            raise TimeoutError
        raise Exception
    except UnicodeError:
        # if a single integer
        if isinstance(_regexp_flags, bool):
            raise TypeError(err_msg)
    except TimeoutError:
        # if is a list of things
        if len(_regexp_flags) != len(_X):
            raise ValueError(err_msg)
        # numbers.Integral covers integers and bool
        if not all(map(isinstance, _regexp_flags,
            ((numbers.Integral, type(None)) for _ in _regexp_flags)
        )):
            raise TypeError(err_msg)
        if any(map(lambda x: x is True, _regexp_flags)):
            raise TypeError(err_msg)
    except Exception as e:
        raise TypeError(err_msg)


    del err_msg




