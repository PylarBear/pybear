# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Sequence
from typing_extensions import Union

import re



def _match_finder(
    _line: list[str],
    _ngram: Sequence[Union[str, re.Pattern]],
) -> list[int]:

    """
    Slide along a sequence of strings looking for matches against an
    n-gram pattern. When one is found, record the first index position
    of the sequence that matches. Sequences cannot overlap.


    Parameters
    ----------
    _line:
        list[str] - A single 1D sequence of strings.
    _ngram:
        Sequence[Union[str, re.Pattern]] - A single n-gram sequence
        containing string literals and/or re.compile objects that
        specify an n-gram pattern. Cannot have less than 2 entries.


    Returns
    -------
    -
        list[int] - the starting indices of sequences that match the
        n-gram pattern.


    """


    # validation wont allow empty ngram
    # but there may be empty lines
    _n_len = len(_ngram)

    if _n_len > len(_line):
        return []


    _hits = []
    _idx = 0
    while _idx + len(_ngram) <= len(_line):

        _block = _line[_idx: _idx + _n_len]

        # _sp = sub_pattern
        _ngram_matches = []
        for _sp, _word in zip(_ngram, _block):

            _ngram_matches.append(
                re.fullmatch(re.escape(_sp) if isinstance(_sp, str) else _sp, _word)
            )

        if all(_ngram_matches):
            _hits.append(_idx)
            _idx += _n_len
        else:
            _idx += 1


    return _hits





