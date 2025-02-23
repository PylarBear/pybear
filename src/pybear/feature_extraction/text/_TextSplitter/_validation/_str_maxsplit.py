# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Sequence
from .._type_aliases import StrMaxSplitType

import numbers



def _val_str_maxsplit(
    _str_maxsplit: StrMaxSplitType,
    _X: Sequence[str]
) -> None:

    """
    Validate the maxsplit parameter for str.split().


    Parameters
    ----------
    _str_maxsplit:
        StrMaxSplitType - the maximum number of splits to perform on each
        string in X, working from left to right. Can be None, an integer,
        or a list of None, integers, or Falses. If None and in str.split()
        mode, the default maxsplits for str.split() is used. If a single
        integer, that integer is applied to all strings in X. If passed
        as a list, the number of integers must equal the number of
        strings in X, and the integers are applied to the corresponding
        string in X. The values are only validated for being integers;
        any exceptions raised beyond that are raised by str.split().

    _X:
        Sequence[str] - the data.


    Return
    ------
    -
        None


    Notes
    -----
    str.split()


    """


    if _str_maxsplit is None:
        return


    err_msg = (
        f"'str_maxsplit' must be None, a single integer, or a list of "
        f"Nones, integers, and Falses whose length equals the number of "
        f"strings in X"
    )


    try:
        if isinstance(_str_maxsplit, numbers.Real):
            raise UnicodeError
        if not isinstance(_str_maxsplit, list):
            raise Exception
        raise TimeoutError
    except UnicodeError:
        # if is a single number
        if not isinstance(_str_maxsplit, numbers.Integral):
            raise TypeError(err_msg)
        if isinstance(_str_maxsplit, bool):
            raise TypeError(err_msg)
    except TimeoutError:
        # if is a list, len must == len(_X) and can contain Nones,
        # integers, or Falses
        if len(_str_maxsplit) != len(_X):
            raise ValueError(err_msg)
        for _ in _str_maxsplit:
            # numbers.Integral covers integers and bool
            if not isinstance(_, (numbers.Integral, type(None))):
                raise TypeError(err_msg)
            if _ is True:
                raise TypeError(err_msg)
    except Exception as e:
        raise TypeError(err_msg)


    del err_msg


