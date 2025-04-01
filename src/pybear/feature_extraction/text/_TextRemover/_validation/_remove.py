# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing_extensions import Union
from .._type_aliases import RemoveType

import numbers
import re



def _val_remove(
    _remove: RemoveType,
    _n_rows: numbers.Integral
) -> None:

    """
    Validate 'remove'.
    Must be:
    None,
    a literal string,
    a regex pattern in a re.compile object,
    a tuple of literal strings and/or regex patterns in re.compile objects,
    or a list of Nones, literal strings, regex patterns in re.compile
    objects, and/or the tuples.

    Regex patterns are not validated here, any exception would be raised
    by re.fullmatch. If passed as a list, the number of entries must
    equal the number of rows in X.


    Parameters
    ----------
    _remove:
        RemoveType - the literal strings or re.compile objects used to
        match patterns for removal from the data. When None, nothing
        is removed. If a single literal or re.compile object, that is
        applied to every row in X. If a tuple of string literals and/or
        re.compile objects, then each of them is applied to every row
        in X. When passed as a list, the number of entries must equal
        the number of rows in X, and the entries are applied to the
        corresponding row in X. The list must be a sequence of Nones,
        string literals, re.compile objects and/or tuples of string
        literals / re.compile objects.
    _n_rows:
        numbers.Integral - the number of rows in the data.


    Return
    ------
    -
        None


    Notes
    -----
    see re.fullmatch()


    """


    assert isinstance(_n_rows, numbers.Integral)
    assert not isinstance(_n_rows, bool)
    assert _n_rows >= 0

    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

    def _val_helper(
        _core_remove: Union[None, str, re.Pattern[str]]
    ) -> bool:
        """
        Helper function to validate the core 'remove' objects are
        Union[None, str, re.Pattern].
        """

        return isinstance(_core_remove, (type(None), str, re.Pattern))


    def _tuple_helper(
        _remove: tuple[Union[str, re.Pattern[str]]]
    ) -> bool:
        """
        Helper function for validating tuples.
        """

        return isinstance(_remove, tuple) \
            and all(map(_val_helper, _remove)) \
            and not any(map(lambda x: x is None, _remove))


    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

    _err_msg = (
        f"'remove' must be None, a literal string, a regex pattern in a "
        f"re.compile object, \na tuple of literal strings and/or regex "
        f"patterns in re.compile objects, or a list of any of those things. "
        f"\nIf passed as a list, the number of entries must equal the "
        f"number of rows in the data."
    )

    if _val_helper(_remove):
        # means is None, str, or re.Pattern
        pass

    elif isinstance(_remove, tuple):

        if not _tuple_helper(_remove):
            raise TypeError(_err_msg)

    elif isinstance(_remove, list):

        if len(_remove) != _n_rows:
            raise ValueError(_err_msg)

        if not all(_val_helper(i) or _tuple_helper(i) for i in _remove):
            raise TypeError(_err_msg)

    else:
        raise TypeError(_err_msg)


    del _val_helper, _err_msg










