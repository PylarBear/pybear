# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing_extensions import TypeAlias, Union

import numbers



FlagType: TypeAlias = Union[None, numbers.Integral]



def _val_flags(
    _flags: Union[FlagType, list[FlagType]],
    _n_rows: numbers.Integral
) -> None:

    """
    Validate the 'flags' parameter. Must be integer, None, or list of
    integer / None. If a list, the length must match the number of rows
    in X.


    Parameters
    ----------
    _flags:
        Union[FlagType, list[FlagType]] - the flags argument for re
        methods/functions when re.Pattern objects are being used (either
        globally on all the data or a row of the data.) Must be None or
        an integer, or a list of Nones and/or integers. When passed as a
        list, the length must match the number of rows in the data. The
        values of the integers are not validated for legitimacy, any
        exceptions would be raised by the re method it is passed to.
    _n_rows:
        numbers.Integral - the number of rows in the data.


    Return
    ------
    -
        None


    """


    assert isinstance(_n_rows, numbers.Integral)
    assert not isinstance(_n_rows, bool)
    assert _n_rows >= 0


    def _val_helper(
        _core_flags: Union[None, numbers.Integral]
    ) -> bool:
        """
        Helper function to validate core flags objects are
        Union[None, int].

        """

        return isinstance(_core_flags, (type(None), numbers.Integral)) and \
                    not isinstance(_core_flags, bool)

    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

    _err_msg = (
        f"'flags' must be None, an integer, or a LIST that contains any "
        f"combination of Nones and/or integers whose length matches the "
        f"number of rows in the data."
    )


    if _val_helper(_flags):
        # means is either None or int
        return

    elif isinstance(_flags, list):

        if len(_flags) != _n_rows:
            raise ValueError(_err_msg)

        if not all(map(_val_helper, _flags)):
            raise TypeError

    else:
        raise TypeError(_err_msg)


    del _val_helper, _err_msg




