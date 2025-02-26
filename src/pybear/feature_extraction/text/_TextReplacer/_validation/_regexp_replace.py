# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Callable
from .._type_aliases import (
    XContainer,
    RegExpReplaceType
)

import numbers
import re



def _val_regexp_replace(
    _regexp_replace: RegExpReplaceType,
    _X: XContainer
) -> None:

    """
    Validate the arguments for re.sub().


    Parameters
    ----------
    _regexp_replace:
        RegExpReplaceType - the arguments for re.sub().
    _X:
        XContainer - the data.


    Returns
    -------
    -
        None


    """

    # coule be:
    # None,
    # tuple(pattern, str/callable), tuple(pattern, str/callable, int),
    # tuple(pattern, str/callable, int, int),
    # set[Union[of the 3 tuples]]
    # list[tuples, set[Union[of the 3 tuples]], Literal[False]]


    # helper function -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    def _args_validation(_tuple: tuple):
        # must have 2, 3, or 4 entries

        """
        Validate the arguments for re.sub() that were passed by the
        user.


        Parameters
        ----------
        _tuple:
            the arguments for re.sub()


        Returns
        -------
        -
            None

        """


        err_msg = (
            f"when passing arguments to 'regexp_replace' for re.sub(), "
            f"there must be 2 - 4 args passed as a tuple. \nThe first must "
            f"always be a string or a re.Pattern object. \nThe second must "
            f"always be a string or a callable. \nThe third is an optional "
            f"integer for number of replacements to make within a string. "
            f"\nThe fourth is an optional re flags integer."
        )

        if not isinstance(_tuple, tuple):
            raise TypeError(err_msg)

        if len(_tuple) not in range(2, 5):
            raise ValueError(err_msg)

        allowed = (
            (str, re.Pattern),
            (str, Callable),
            numbers.Integral,
            numbers.Integral
        )

        for _idx, _arg in enumerate(_tuple):

            if not isinstance(_arg, allowed[_idx]):
                raise TypeError(err_msg)

    # END helper function -- -- -- -- -- -- -- -- -- -- -- -- -- -- --


    err_msg = (
        f"'regexp_replace' must be None, a tuple, a python set of tuples, "
        f"or a python list containing tuples, python sets of tuples, or "
        f"Falses. \nsee the docs for more details."
    )


    if _regexp_replace is None:
        return
    elif isinstance(_regexp_replace, tuple):
        _args_validation(_regexp_replace)
    elif isinstance(_regexp_replace, set):
        for _tuple in _regexp_replace:
            _args_validation(_tuple)
    elif isinstance(_regexp_replace, list):
        if len(_regexp_replace) != len(_X):
            raise ValueError(
                f"if 'regexp_replace' is passed as a list its length "
                f"must equal the length of X"
            )
        for _row in _regexp_replace:
            if _row is False:
                continue
            elif isinstance(_row, tuple):
                _args_validation(_row)
            elif isinstance(_row, set):
                for _tuple in _row:
                    _args_validation(_tuple)
            else:
                raise TypeError(err_msg)
    else:
        raise TypeError(err_msg)








