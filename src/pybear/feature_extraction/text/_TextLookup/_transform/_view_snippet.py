# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Sequence

import math
import numbers



def _view_snippet(
    _VECTOR: Sequence[str],
    _idx: numbers.Integral,
    _span: numbers.Integral = 9
) -> str:

    """
    Highlights the word of interest (which is given by the 'idx' value)
    in a series of words. For example, in a simple case that avoids edge
    effects, a span of 9 would show 4 strings to the left of the target
    string in lower-case, the target string itself capitalized, then the
    4 strings to the right of the target string in lower-case.


    Parameters
    ----------
    _VECTOR:
        Sequence[str] - the sequence of strings that provides a
        subsequence of strings to highlight. Cannot be empty.
    _idx:
        numbers.Integral - the index of the string in the sequence of
        strings to highlight.
    _span:
        Optional[numbers.Integral], default = 9 - the number of strings
        in the sequence of strings to select when highlighting one
        particular central string.


    Return
    ------
    -
        str: the highlighted portion of the string sequence.


    """

    # validation -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    if not isinstance(_VECTOR, list):
        raise TypeError(f"expected VECTOR to be a list")
    if not all(map(isinstance, _VECTOR, (str for _ in _VECTOR))):
        raise TypeError(f"expected VECTOR to contain strings")

    if not isinstance(_idx, numbers.Integral):
        raise TypeError(f"expected idx to be an integer")
    if not _idx >= 0:
        raise TypeError(f"idx must be a positive integer")

    if not isinstance(_span, numbers.Integral):
        raise TypeError(f"expected span to be an integer")
    if not _span >= 0:
        raise TypeError(f"span must be a positive integer")

    # END validation -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --


    _lower = math.floor(_idx - (_span - 1) / 2)
    _upper = math.ceil(_idx + (_span - 1) / 2)
    if _lower <= 0:
        _min, _max = 0, min(_span, len(_VECTOR))
    elif _upper >= len(_VECTOR):
        _min, _max = max(0, len(_VECTOR) - _span), len(_VECTOR)
    else:
        _min, _max = _lower, _upper + 1
    del _lower, _upper

    SNIPPET = []
    for word_idx in range(_min, _max):
        if word_idx == _idx:
            SNIPPET.append(_VECTOR[word_idx].upper())
        else:  # word_idx is not on the target word...
            SNIPPET.append(_VECTOR[word_idx].lower())


    return " ".join(SNIPPET)  # RETURNS AS STRING




