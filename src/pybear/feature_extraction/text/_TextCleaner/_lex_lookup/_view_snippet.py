# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

import math
from typing_extensions import Union
import numpy as np



def _view_snippet(
        VECTOR: Union[list[str], np.ndarray[str]],
        idx: int,
        span: int=9
    ) -> str:

    """
    Highlights the word of interest in a sub-series of words.

    Parameters
    ----------
    VECTOR:
        Union[list[str], np.ndarray[str]] - Vector of words
    idx:
        int - index position to view
    span:
        int - span of words to view around the word in the idx position

    Return
    ------
    SNIPPET
        SNIPPET: str - view of the target word and surrounding words


    """

    err_msg = f"'VECTOR' must be a list-like vector of individual words"

    try:
        iter(VECTOR)
        if isinstance(VECTOR, (str, dict)):
            raise Exception
        VECTOR = np.array(VECTOR, dtype=object)
    except:
        raise ValueError(err_msg)

    if len(VECTOR.shape) > 1:
        raise ValueError(err_msg)

    if not all(map(isinstance, VECTOR, (str for _ in VECTOR))):
        raise ValueError(err_msg)

    del err_msg

    try:
        float(idx)
        if isinstance(idx, bool):
            raise Exception
        if int(idx) != idx:
            raise Exception
        if idx not in range(0, len(VECTOR)):
            raise Exception
    except:
        raise ValueError(
            f'idx MUST BE A NON-NEGATIVE INTEGER IN RANGE OF GIVEN VECTOR'
        )


    try:
        float(span)
        if isinstance(span, bool):
            raise Exception
        if not span > 0:
            raise Exception
        if int(span) != span:
            raise Exception
    except:
        raise ValueError(
            f'span MUST BE A NON-NEGATIVE INTEGER IN RANGE OF GIVEN VECTOR'
        )

    _lower = math.floor(idx - (span - 1) / 2)
    _upper = math.ceil(idx + (span - 1) / 2)

    if _lower <= 0:
        _min = 0
        _max = min(span, len(VECTOR))
    elif _upper >= len(VECTOR):
        _min = max(0, len(VECTOR) - span)
        _max = len(VECTOR)
    else:
        _min = _lower
        _max = _upper
    del _lower, _upper

    SNIPPET = []
    for word_idx in range(_min, _max):
        if word_idx == idx:  # majuscule the target word...
            SNIPPET.append(VECTOR[word_idx].upper())
        else:  # word_idx is not on the target word...
            SNIPPET.append(VECTOR[word_idx].lower())

    SNIPPET = " ".join(SNIPPET)  # RETURNS AS STRING

    return SNIPPET









