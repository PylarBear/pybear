# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from .._type_aliases import (
    XContainer,
    StrReplaceType
)

import numbers




def _val_str_replace(
    _str_replace: StrReplaceType,
    _X: XContainer
) -> None:

    """
    Validate the arguments for str.replace().


    Parameters
    ----------
    _str_replace:
        StrReplaceType - the arguments for str.replace().
    _X:
        XContainer - the data.


    Returns
    -------
    -
        None


    """

    # coule be:
    # None,
    # tuple(str, str), tuple(str, str, int),
    # set[Union[of the 2 tuples]]
    # list[Union[tuples, set[Union[of the 2 tuples]], Literal[False]]]


    # helper function -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    def _args_validation(_tuple: tuple):
        # must have 2 or 3 entries

        """
        Validate the arguments for str.replace() that were passed by the
        user.


        Parameters
        ----------
        _tuple:
            tuple - the arguments for str.replace()


        Returns
        -------
        -
            None

        """

        err_msg = (
            f"when passing arguments to 'str_replace' for str.replace(), "
            f"there must be 2 or 3 args passed as a tuple - (str, str) "
            f"or (str, str, int)"
        )

        if not isinstance(_tuple, tuple):
            raise TypeError(err_msg)

        if len(_tuple) not in [2, 3]:
            raise ValueError(err_msg)

        allowed = (str, str, numbers.Integral)

        for _idx, _arg in enumerate(_tuple):

            if not isinstance(_arg, allowed[_idx]):
                raise TypeError(err_msg)
    # END helper function -- -- -- -- -- -- -- -- -- -- -- -- -- -- --


    err_msg = (
        f"'str_replace' must be None, a tuple, a python set of tuples, "
        f"or a python list containing tuples, python sets of tuples, or "
        f"Falses. \nsee the docs for more details."
    )


    if _str_replace is None:
        return
    elif isinstance(_str_replace, tuple):
        _args_validation(_str_replace)
    elif isinstance(_str_replace, set):
        for _tuple in _str_replace:
            _args_validation(_tuple)
    elif isinstance(_str_replace, list):
        if len(_str_replace) != len(_X):
            raise ValueError(
                f"if 'str_replace' is passed as a list its length "
                f"must equal the length of X"
            )
        for _row in _str_replace:
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








