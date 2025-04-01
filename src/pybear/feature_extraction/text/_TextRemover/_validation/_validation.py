# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from .._type_aliases import (
    XContainer,
    RemoveType,
    CaseSensitiveType,
    RemoveEmptyRowsType,
    FlagsType
)

from ._remove import _val_remove
from ._case_sensitive import _val_case_sensitive
from ._remove_empty_rows import _val_remove_empty_rows
from ._flags import _val_flags

from .....base._check_1D_str_sequence import check_1D_str_sequence
from .....base._check_2D_str_array import check_2D_str_array



def _validation(
    _X: XContainer,
    _remove: RemoveType,
    _case_sensitive: CaseSensitiveType,
    _remove_empty_rows: RemoveEmptyRowsType,
    _flags: FlagsType
) -> None:


    """
    Centralized hub for validation. See the individual modules for more
    details.

    Beyond the individual modules' validation, this module also checks:
    1) cannot pass anything to 'flags' if 'remove' is None
    2) cannot pass a list to 'case_sensitive' if 'remove' is None


    Parameters:
    -----------
    _X:
        XContainer - the data. 1D or (possible ragged) 2D array of
        strings.
    _remove:
        RemoveType - the string literals or re.compile patterns to look
        for and remove.
    _case_sensitive:
        CaseSensitiveType - whether to make searches for strings to
        remove case-sensitive.
    _remove_empty_rows:
        RemoveEmptyRowsType - whether to remove empty rows from 2D data.
        This does not apply to 1D data, by definition rows will always
        be removed from 1D data.
    _flags:
        FlagsType - externally provided flags if using re.compile objects.


    Return
    ------
    -
        None


    """


    try:
        check_2D_str_array(_X, require_all_finite=False)

        # remove_empty_rows only applies to 2D data
        _val_remove_empty_rows(_remove_empty_rows)
    except Exception as e:
        try:
            check_1D_str_sequence(_X, require_all_finite=False)
        except Exception as f:
            raise TypeError(
                f"Expected X to be 1D sequence or (possibly ragged) 2D "
                f"array of string-like values."
            )


    _n_rows = _X.shape[0] if hasattr(_X, 'shape') else len(_X)

    _val_remove(_remove, _n_rows)

    _val_case_sensitive(_case_sensitive, _n_rows)

    _val_flags(_flags, _n_rows)

    del _n_rows

    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

    #########
    if _remove is None and isinstance(_case_sensitive, list):
        raise ValueError(
            f"cannot pass 'case_sensitive' as a list if 'remove' is not "
            f"passed."
        )

    #########
    if _remove is None and _flags is not None:
        raise ValueError(f"cannot pass 'flags' if 'remove' is not passed.")












