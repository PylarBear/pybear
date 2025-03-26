# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Callable, Sequence
from typing_extensions import Union
import numpy.typing as npt

import re

import numpy as np

from ._match_finder import _match_finder
from ._replacer import _replacer
from ._wrap_manager import _wrap_manager



def _transform(
    _X: list[list[str]],
    _ngrams: Sequence[Sequence[Union[str, re.Pattern]]],
    _ngcallable: Union[Callable[[Sequence[str]], str], None],
    _sep: Union[str, None],
    _wrap: bool,
    _remove_empty_rows: bool
) -> tuple[list[list[str]], npt.NDArray[bool]]:

    """
    Scan over the full dataset looking for patterns that match the given
    ngrams. When there is a match, merge the words into a single
    contiguous string mapped from the words.

    Merge ngrams that match ngram patterns using the following hierarchy:

    by the given callable

    by the given separator

    by the default separator


    Parameters
    ----------
    _X:
        list[list[str]] - (possibly ragged) 2D array of strings.
    _ngrams:
        _ngrams: Sequence[Sequence[Union[str, re.Pattern]] - The n-gram
        sequences to look for in the data. Each individual n-gram must
        be a sequence of string literals and/or re.Pattern objects that
        specify an n-gram pattern. Cannot be empty, and cannot have any
        ngrams that have less than 2 entries.
    _ngcallable:
        Union[Callable[[Sequence[str]], str], None] - the callable
        applied to sequences that match an n-gram pattern to produce a
        single contiguous string.
    _sep:
        Union[str, None] - the user defined separator to join the words
        with, if _ngcallable is not given. If no separator is defined by
        the user, use the default separator.
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
        list[list[str]] - the data with all matching n-gram patterns
        replaced with contiguous strings.

        NDArray[bool] - a 1D boolean vector of shape (n_rows_, ) that
        indicates what rows of the data were kept (True) and what rows
        were removed (False)  during transform. A row can only be removed
        from the data if 'wrap' is True.

    """


    # need to do ngrams from longest to shortest.

    _lens = list(reversed(list(set(map(len, _ngrams)))))

    for _len in _lens:

        for _ngram in _ngrams:

            if len(_ngram) != _len:
                continue

            for _row_idx in range(len(_X)):

                _hits = _match_finder(_X[_row_idx], _ngram)

                if _wrap and _row_idx < len(_X) - 1:
                    _X[_row_idx], _X[_row_idx+1] = _wrap_manager(
                        _X[_row_idx],
                        _X[_row_idx+1],
                        _hits,
                        [],
                        _ngram,
                        _ngcallable,
                        _sep
                    )

                _X[_row_idx] = _replacer(
                    _X[_row_idx], _ngram, _hits, _ngcallable, _sep
                )

    _row_support = np.ones((len(_X), ), dtype=bool)
    if _remove_empty_rows:
        for _row_idx in range(len(_X)-1, -1, -1):
            if len(_X[_row_idx]) == 0:
                _X.pop(_row_idx)
                _row_support[_row_idx] = False


    return _X, _row_support




