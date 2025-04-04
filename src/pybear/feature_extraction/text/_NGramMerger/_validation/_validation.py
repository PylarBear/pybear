# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Callable, Sequence
from typing_extensions import Union
from .._type_aliases import XContainer

import re

from ._ngrams import _val_ngrams

from ._ngcallable import _val_ngcallable

from ...__shared._validation._2D_X import _val_2D_X
from ...__shared._validation._any_bool import _val_any_bool
from ...__shared._validation._any_string import _val_any_string



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
        literals and/or re.compile objects that specify an n-gram.
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


    _val_2D_X(_X, _require_all_finite=True)

    _val_ngrams(_ngrams)

    _val_ngcallable(_ngcallable)

    _val_any_string(_sep, 'sep', _can_be_None=True)

    _val_any_bool(_wrap, 'wrap')

    _val_any_bool(_remove_empty_rows, 'remove_empty_rows')






