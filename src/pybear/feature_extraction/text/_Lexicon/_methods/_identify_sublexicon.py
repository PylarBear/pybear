# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Sequence
from typing_extensions import Union

import numpy as np

from ...alphanumeric_str import alphabet_str
from ._validate_word_input import _validate_word_input



def _identify_sublexicon(
    WORDS: Union[str, Sequence[str]]
) -> list[str]:

    """
    Identify the files that need to be accessed to make changes to the
    pybear lexicon. These should be found by the first letter of the
    word(s) in WORDS.


    Parameters
    ----------
    WORDS:
        Union[str, Sequence[str]] - the word or sequence of words passed
        to a pybear Lexicon method.


    """


    _validate_word_input(
        WORDS,
        character_validation=False,
        majuscule_validation=False
    )


    if isinstance(WORDS, str):
        _WORDS = [WORDS]
    else:
        _WORDS = WORDS


    _unq_first_chars = np.unique(list(map(lambda x: x[0], _WORDS)))

    _unq_first_chars = sorted(list(map(str.lower, _unq_first_chars)))

    for _char in _unq_first_chars:

        if _char not in alphabet_str():
            raise ValueError(
                f"when looking for sublexicons to update, all first "
                f"characters of words must be one of the 26 letters in "
                f"the English alphabet. Got {_char}."
            )


    return _unq_first_chars













