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

from ._sep import _val_sep

from ._ngcallable import _val_ngcallable

from .....base._check_2D_str_array import check_2D_str_array



def _validation(
    _X: XContainer,
    _ngrams: Sequence[Sequence[Union[str, re.Pattern]]],
    _ngcallable: Union[Callable[[Sequence[str]], str], None],
    _sep: Union[str, None]
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
        Cannot be empty.
    _ngcallable:
        Union[Callable[[Sequence[str]], str], None] - the callable
        applied to ngram sequences to produce a contiguous string
        sequence.
    _sep:
        Union[str, None] - the separator that joins words in the n-grams.


    Returns
    -------
    -
        None


    """


    check_2D_str_array(_X, require_all_finite=True)

    _val_ngrams(_ngrams)

    _val_ngcallable(_ngcallable)

    _val_sep(_sep)








