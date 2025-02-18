# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Sequence

import math
import numbers

from ..._shared._validation._1D_str_sequence import _val_1D_str_sequence



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

    # validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    _val_1D_str_sequence(VECTOR)
    if len(VECTOR) == 0:
        raise ValueError(f"'VECTOR' cannot be empty")


    # idx -- -- -- -- -- -- -- -- -- --
    err_msg = f"'idx' must be a non-negative integer in range of the given vector"
    if not isinstance(idx, numbers.Integral):
        raise TypeError(err_msg)
    if isinstance(idx, bool):
        raise TypeError(err_msg)
    if idx not in range(0, len(VECTOR)):
        raise ValueError(err_msg)
    del err_msg
    # END idx -- -- -- -- -- -- -- -- --

    # span -- -- -- -- -- -- -- -- -- --
    err_msg = f"'span' must be an integer > 3"
    if not isinstance(span, numbers.Integral):
        raise TypeError(err_msg)
    if isinstance(span, bool):
        raise TypeError(err_msg)
    if span < 3:
        raise ValueError(err_msg)
    del err_msg
    # END span -- -- -- -- -- -- -- -- --

    # END validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *



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




