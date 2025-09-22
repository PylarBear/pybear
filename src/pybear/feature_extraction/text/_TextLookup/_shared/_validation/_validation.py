# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Sequence

import re

import numpy as np

from ._delete_always import _val_delete_always
from ._replace_always import _val_replace_always
from ._skip_always import _val_skip_always
from ._split_always import _val_split_always

from ....__shared._validation._2D_X import _val_2D_X
from ....__shared._validation._any_bool import _val_any_bool



def _validation(
    _X,
    _update_lexicon: bool,
    _skip_numbers: bool,
    _auto_split: bool,
    _auto_add_to_lexicon: bool,
    _auto_delete: bool,
    _DELETE_ALWAYS: None | Sequence[str | re.Pattern[str]],
    _REPLACE_ALWAYS: None | dict[str | re.Pattern[str], str],
    _SKIP_ALWAYS: None | Sequence[str | re.Pattern[str]],
    _SPLIT_ALWAYS: None | dict[str | re.Pattern[str], Sequence[str]],
    _remove_empty_rows: bool,
    _verbose: bool
) -> None:
    """Validate `TextLookup` parameters.

    This is a centralized hub for validation. The brunt of the validation
    is handled by the individual modules. See their docs for more
    details.

    Beyond the basic validation of the parameters, manage the
    interdependency of parameters.

    `SKIP_ALWAYS`, `SPLIT_ALWAYS`, `DELETE_ALWAYS`, `REPLACE_ALWAYS`
    must not have common strings (case_sensitive). Conflict is checked
    for any strings in these four objects, not for re.compile objects.
    It is impossible to validate re.compile objects for conflict unless
    you have the actual text that they will be searched against.

    Parameters
    ----------
    _X: XContainer
    _update_lexicon : bool
    _skip_numbers : bool
    _auto_split : bool
    _auto_add_to_lexicon : bool
    _auto_delete : bool
    _DELETE_ALWAYS : None | Sequence[str | re.Pattern[str]]
    _REPLACE_ALWAYS : None | dict[str | re.Pattern[str], str]
    _SKIP_ALWAYS : None | Sequence[str | re.Pattern[str]]
    _SPLIT_ALWAYS : None | dict[str | re.Pattern[str], Sequence[str]]
    _remove_empty_rows : bool
    _verbose : bool

    Returns
    -------
    None

    """


    _val_2D_X(_X, _require_all_finite=False)

    _val_any_bool(_update_lexicon, 'update_lexicon')

    _val_any_bool(_skip_numbers, 'skip_numbers')

    _val_any_bool(_auto_split, 'auto_split')

    _val_any_bool(_auto_add_to_lexicon, 'auto_add_to_lexicon')

    _val_any_bool(_auto_delete, 'auto_delete')

    _val_delete_always(_DELETE_ALWAYS)

    _val_any_bool(_remove_empty_rows, 'remove_empty_rows')

    _val_replace_always(_REPLACE_ALWAYS)

    _val_skip_always(_SKIP_ALWAYS)

    _val_split_always(_SPLIT_ALWAYS)

    _val_any_bool(_verbose, 'verbose')


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

    # DELETE_ALWAYS: None | Sequence[str | re.Pattern[str]] = None
    # REPLACE_ALWAYS: None | dict[str | re.Pattern[str], str] = None
    # SKIP_ALWAYS: None | Sequence[str | re.Pattern[str]] = None
    # SPLIT_ALWAYS: None | dict[str | re.Pattern[str], Sequence[str]] = None

    # condition the containers to check for conflict between any strings
    # create WIP objects that only have strings, remove any re.compile objects

    delete_always = []
    if _DELETE_ALWAYS:   # if what was passed is not None
        # we know from _val_delete_always that it must contain str or re.compile
        delete_always = [x for x in _DELETE_ALWAYS if isinstance(x, str)]
    # but if what was passed was None, then delete_always stays []

    replace_always_keys = []
    if _REPLACE_ALWAYS:   # if what was passed is not None
        # we know from _val_replace_always that keys must be str or re.compile
        replace_always_keys = [x for x in _REPLACE_ALWAYS if isinstance(x, str)]
    # but if what was passed was None, then replace_always_keys stays []

    skip_always = []
    if _SKIP_ALWAYS:   # if what was passed is not None
        # we know from _val_skip_always that it must contain str or re.compile
        skip_always = [x for x in _SKIP_ALWAYS if isinstance(x, str)]
    # but if what was passed was None, then skip_always stays []

    split_always_keys = []
    if _SPLIT_ALWAYS:   # if what was passed is not None
        # we know from _val_split_always that keys must be str or re.compile
        split_always_keys = [x for x in _SPLIT_ALWAYS if isinstance(x, str)]
    # but if what was passed was None, then split_always_keys stays []


    ALL = np.hstack((
        delete_always,
        replace_always_keys,
        skip_always,
        split_always_keys
    )).tolist()

    if not np.array_equal(sorted(list(set(ALL))), sorted(ALL)):

        # if there are no duplicates among the strings in ALL, then set(ALL)==ALL

        UNQS, CTS = np.unique(ALL, return_counts=True)

        UNQ_CT_DICT = dict((zip(
            list(map(str, UNQS)),
            list(map(int, CTS))
        )))

        UNQ_CT_DICT = {k:v for k, v in UNQ_CT_DICT.items() if v >= 2}

        raise ValueError(
            f"{', '.join(UNQ_CT_DICT)} appear more than once in the specially "
            f"handled words."
        )





