# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Sequence
from .._type_aliases import RegExpFlagsType

import numbers



def _val_regexp_flags(
    _regexp_flags: RegExpFlagsType,
    _X: Sequence[str]
) -> None:

    """
    Validate the 'flags' parameter for re.split(). Must be integer, None,
    or list of integer, None, and literal False.


    Parameters
    ----------
    _flags:
         RexExpFlagsType - the flags arguments for re.split(), if regular
         expressions are being used. Must be None, an integer, or a list
         of Nones, integers, or Falses. When passed as a list, the
         length must match the number of strings in the data. The values
         of the integers are not validated for legitimacy, any exceptions
         would be raised by re.split().


    Return
    ------
    -
        None


    Notes
    -----
    re.split()

    """


    if _regexp_flags is None:
        return


    err_msg = (
        f"'regexp_flags' must be None, a single integer, or a list that "
        f"contains any combination of Nones, integers, or literal Falses, "
        f"whose length matches the number of strings in X."
    )

    try:
        if isinstance(_regexp_flags, numbers.Integral):
            raise UnicodeError
        if not isinstance(_regexp_flags, list):
            raise Exception
        if len(_regexp_flags) != len(_X):
            raise TimeoutError
        for _ in _regexp_flags:
            # numbers.Integral covers integers and bool
            if not isinstance(_, (numbers.Integral, type(None))):
                raise Exception
            if _ is True:
                raise Exception
    except UnicodeError:
        # if a single integer
        if isinstance(_regexp_flags, bool):
            raise TypeError(err_msg)
    except TimeoutError:
        raise ValueError(err_msg)
    except Exception as e:
        raise TypeError(err_msg)







