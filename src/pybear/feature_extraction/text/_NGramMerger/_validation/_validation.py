# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Callable, Sequence
from typing_extensions import Union
from .._type_aliases import XContainer

import re

from .....base._check_2D_str_array import check_2D_str_array

from ._ngrams import _val_ngrams

from ._sep import _val_sep

from ._ngcallable import _val_ngcallable

from ._wrap import _val_wrap

from ._remove_empty_rows import _val_remove_empty_rows



def _validation(
    _X: XContainer,
    _ngrams: Sequence[Sequence[Union[str, re.Pattern]]],
    _ngcallable: Union[Callable[[Sequence[str]], str], None],
    _sep: Union[str, None],
    _wrap: bool,
    _remove_empty_rows: bool
) -> None:

    """
    Centralized hub for validation. See the individual modules for
    more details.


    Parameters
    ----------
    _X:
        XContainer - (possibly ragged) 2D array of strings.
    _ngrams:
        Sequence[Sequence[Union[str, re.Pattern]]] - A sequence of
        sequences, where each inner sequence holds a series of string
        literals and/or re.Pattern objects that specify an n-gram.
        Cannot be empty, and cannot have any n-grams with less than 2
        entries.
    _ngcallable:
        Union[Callable[[Sequence[str]], str], None] - the callable
        applied to ngram sequences to produce a contiguous string
        sequence.
    _sep:
        Union[str, None] - the separator that joins words in the n-grams.
    _wrap:
        bool - whether to look for pattern matches across the end of the
        current line and beginning of the next line.
    _remove_empty_rows:
        bool - whether to delete any empty rows that may occur during
        the merging process. A row could only become empty if 'wrap' is
        True.


    Returns
    -------
    -
        None


    """


    check_2D_str_array(_X, require_all_finite=True)

    _val_ngrams(_ngrams)

    _val_ngcallable(_ngcallable)

    _val_sep(_sep)

    _val_wrap(_wrap)

    _val_remove_empty_rows(_remove_empty_rows)



