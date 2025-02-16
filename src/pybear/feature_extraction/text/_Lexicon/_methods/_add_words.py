# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Sequence, Optional
from typing_extensions import Union

import os

import numpy as np

from ._validate_word_input import _validate_word_input
from ._identify_sublexicon import _identify_sublexicon


# MODULE FOR APPENDING NEW WORDS TO A SUB-LEXICON


def _add_words(
    WORDS: Union[str, Sequence[str]],
    lexicon_folder_path: str,
    character_validation: Optional[bool] = True,
    majuscule_validation: Optional[bool] = True
) -> None:

    """
    Update the pybear lexicon with the given words.


    Parameter
    ---------
    WORDS:
        Union[str, Sequence[str]] - the word or words to be appended to
        the pybear lexicon. Cannot be an empty string or an empty
        sequence.
    lexicon_folder_path:
        str - the path to the directory that holds the lexicon text
        files.
    character_validation:
        Optional[bool], default = True - whether to apply pybear lexicon
        character validation to the word or sequence of words. pybear
        lexicon allows only the 26 letters in the English language, no
        others. No spaces, no hypens, no apostrophes. If True, any
        non-alpha characters will raise an exception during validation.
        If False, any string character is accepted.
    majuscule_validation:
        Optional[bool], default = True - whether to apply pybear lexicon
        majuscule validation to the word or sequence of words. The pybear
        lexicon requires all characters be majuscule, i.e., EVERYTHING
        MUST BE UPPER-CASE. If True, any non-majuscule characters will
        raise an exception during validation. If False, any case is
        accepted.


    Return
    ------
    -
        None

    """


    _validate_word_input(WORDS)

    if not isinstance(lexicon_folder_path, str):
        raise TypeError(f"'lexicon_folder_path' must be a string")

    if not isinstance(character_validation, bool):
        raise TypeError(f"'character_validation' must be boolean")

    if not isinstance(majuscule_validation, bool):
        raise TypeError(f"'majuscule_validation' must be boolean")
    # END validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    file_base = f'lexicon_'

    file_identifiers: list[str] = _identify_sublexicon(WORDS)

    for file_letter in file_identifiers:

        full_path = os.path.join(
            lexicon_folder_path,
            file_base + file_letter.lower() + '.txt'
        )

        with open(full_path, 'r') as f:
            raw_text = np.fromiter(f, dtype='<U40')

        OLD_SUB_LEXICON = np.char.replace(raw_text, f'\n', f' ')
        del raw_text

        PERTINENT_WORDS = [w for w in WORDS if w[0].lower() == file_letter]

        NEW_LEXICON = np.hstack((OLD_SUB_LEXICON, PERTINENT_WORDS))

        # MUST USE uniques TO TAKE OUT ANY NEW WORDS ALREADY IN LEXICON (AND SORT)
        NEW_LEXICON = np.unique(NEW_LEXICON)


        with open(full_path, 'w') as f:
            for line in NEW_LEXICON:
                f.write(line+f'\n')
            f.close()


    print(f'\n*** Lexicon update successful. ***\n')










