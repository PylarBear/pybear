# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Sequence

import math
import numbers

from ._validation._view_snippet import _view_snippet_validation



def _view_snippet(
    VECTOR: Sequence[str],
    idx: numbers.Integral,
    span: numbers.Integral = 9
) -> str:

    """
    Highlights the word of interest (which is given by the 'idx' value)
    in a series of words. For example, in a simple case that avoids edge
    effects, a span of 9 would show 4 strings to the left of the target
    string in lower-case, the target string itself capitalized, then the
    4 strings to the right of the target string in lower-case.


    Parameters
    ----------
    VECTOR:
        Sequence[str] - the sequence of strings that provides a
        subsequence of strings to highlight. Cannot be empty.
    idx:
        numbers.Integral - the index of the string in the sequence of
        strings to highlight.
    span:
        Optional[numbers.Integral], default = 9 - the number of strings
        in the sequence of strings to select when highlighting one
        particular central string.


    Return
    ------
    -
        str: the highlighted portion of the string sequence.


    """


    _view_snippet_validation(VECTOR, idx, span)


    _lower = math.floor(idx - (span - 1) / 2)
    _upper = math.ceil(idx + (span - 1) / 2)
    if _lower <= 0:
        _min, _max = 0, min(span, len(VECTOR))
    elif _upper >= len(VECTOR):
        _min, _max = max(0, len(VECTOR) - span), len(VECTOR)
    else:
        _min, _max = _lower, _upper + 1
    del _lower, _upper

    SNIPPET = []
    for word_idx in range(_min, _max):
        if word_idx == idx:
            SNIPPET.append(VECTOR[word_idx].upper())
        else:  # word_idx is not on the target word...
            SNIPPET.append(VECTOR[word_idx].lower())


    return " ".join(SNIPPET)  # RETURNS AS STRING




