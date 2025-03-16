# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Sequence

import numpy as np

from ._update_lexicon import _val_update_lexicon
from ._skip_numbers import _val_skip_numbers
from ._auto_split import _val_auto_split
from ._auto_add_to_lexicon import _val_auto_add_to_lexicon
from ._auto_delete import _val_auto_delete
from ._delete_always import _val_delete_always
from ._replace_always import _val_replace_always
from ._skip_always import _val_skip_always
from ._split_always import _val_split_always
from ._verbose import _val_verbose

from .....base._check_2D_str_array import check_2D_str_array



def _validation(
    _X,
    _update_lexicon: bool,
    _skip_numbers: bool,
    _auto_split: bool,
    _auto_add_to_lexicon: bool,
    _auto_delete: bool,
    _DELETE_ALWAYS: Sequence[str],
    _REPLACE_ALWAYS:dict[str, str],
    _SKIP_ALWAYS: Sequence[str],
    _SPLIT_ALWAYS: dict[str, Sequence[str]],
    _verbose: bool
) -> None:

    """
    Validate TextLookup parameters. This is a centralized hub for
    validation. The brunt of the validation is handled by the individual
    modules. See their docs for more details.

    Manage the interdependency of parameters.

    SKIP_ALWAYS, SPLIT_ALWAYS, DELETE_ALWAYS, REPLACE_ALWAYS must not
    have common strings (case_sensitive).


    Parameters
    ----------
    _X: XContainer
    _update_lexicon: bool
    _skip_numbers: bool
    _auto_split: bool
    _auto_add_to_lexicon: bool
    _auto_delete: bool
    _DELETE_ALWAYS: Sequence[str]
    _REPLACE_ALWAYS:dict[str, str]
    _SKIP_ALWAYS: Sequence[str]
    _SPLIT_ALWAYS: dict[str, Sequence[str]]
    _verbose: bool


    Return
    ------
    -
        None


    """


    check_2D_str_array(_X, require_all_finite=True)

    _val_update_lexicon(_update_lexicon)

    _val_skip_numbers(_skip_numbers)

    _val_auto_split(_auto_split)

    _val_auto_add_to_lexicon(_auto_add_to_lexicon)

    _val_auto_delete(_auto_delete)

    _val_delete_always(_DELETE_ALWAYS)

    _val_replace_always(_REPLACE_ALWAYS)

    _val_skip_always(_SKIP_ALWAYS)

    _val_split_always(_SPLIT_ALWAYS)

    _val_verbose(_verbose)


    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --



    if _auto_add_to_lexicon and not _update_lexicon:
        raise ValueError(
            f"'auto_add_to_lexicon' cannot be True if 'update_lexicon' is False"
        )

    if _update_lexicon and _auto_delete:
        raise ValueError(
            f"'update_lexicon' and 'auto_delete' cannot be True simultaneously"
        )



    # SKIP_ALWAYS, SPLIT_ALWAYS, DELETE_ALWAYS, REPLACE_ALWAYS must not
    # have common strings (case_sensitive).

    # DELETE_ALWAYS: Optional[Union[Sequence[str], None]] = None,
    # REPLACE_ALWAYS: Optional[Union[dict[str, str], None]] = None,
    # SKIP_ALWAYS: Optional[Union[Sequence[str], None]] = None,
    # SPLIT_ALWAYS: Optional[Union[dict[str, Sequence[str]], None]] = None,

    delete_always = list(_DELETE_ALWAYS) if _DELETE_ALWAYS else []
    replace_always_keys = list(_REPLACE_ALWAYS.keys()) if _REPLACE_ALWAYS else []
    skip_always = list(_SKIP_ALWAYS) if _SKIP_ALWAYS else []
    split_always_keys = list(_SPLIT_ALWAYS.keys()) if _SPLIT_ALWAYS else []

    ALL = np.hstack((
        delete_always,
        replace_always_keys,
        skip_always,
        split_always_keys
    )).tolist()

    if not np.array_equal(sorted(list(set(ALL))), sorted(ALL)):

        _ = np.unique(ALL, return_counts=True)
        __ = list(map(str, [k for k, v in zip(*_) if v >= 2]))

        raise ValueError(
            f"{', '.join(__)} appear more than once in the specially handled words."
        )


    del ALL




