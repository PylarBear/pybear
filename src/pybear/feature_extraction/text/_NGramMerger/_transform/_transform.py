# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Callable, Sequence
from typing_extensions import Union

import re

from ._match_finder import _match_finder
from ._replacer import _replacer



def _transform(
    _X: list[list[str]],
    _ngrams: Sequence[Sequence[Union[str, re.Pattern]]],
    _ngcallable: Union[Callable[[Sequence[str]], str], None],
    _sep: Union[str, None],
    _wrap: bool
) -> list[list[str]]:

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
        specify an n-gram pattern. Cannot be empty.
    _ngcallable:
        Union[Callable[[Sequence[str]], str], None] - the callable
        applied to sequences that match an n-gram pattern to produce a
        single contiguous string.
    _sep:
        Union[str, None] - the user defined separator to join the words
        with, if _ngcallable is not given. If no separator is defined by
        the user, use the default separator.
    _wrap:
        bool - whether to look for pattern matches across the ends and
        beginnings of two adjacent lines.


    Returns
    -------
    -
        list[list[str]] - the data with all matching n-gram patterns
        replaced with contiguous strings.


    """


    # need to do ngrams from longest to shortest.

    _lens = list(reversed(list(set(map(len, _ngrams)))))

    for _len in _lens:

        for _ngram in _ngrams:

            if len(_ngram) != _len:
                continue

            for _row_idx in range(len(_X)-1, -1, -1):

                # if _wrap and _row_idx < len(_X)-1:
                #     _n_len = len(_ngram)
                #     _WRAPPED = _X[_row_idx][-_n_len+1:]
                #     _WRAPPED += _X[_row_idx+1][:_n_len-1]
                #     _NEW_WRAPPED = _slider(_WRAPPED, _ngram, _ngcallable, _sep)
                #     if len(_NEW_WRAPPED) != len(_WRAPPED):
                #         # then there was a merge

                _hits = _match_finder(_X[_row_idx], _ngram)
                _X[_row_idx] = _replacer(
                    _X[_row_idx], _ngram, _hits, _ngcallable, _sep
                )
                _previous_hits = _hits

    return _X




