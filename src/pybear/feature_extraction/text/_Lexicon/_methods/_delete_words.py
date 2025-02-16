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




def _delete_words(
    WORDS: Union[str, Sequence[str]],
    lexicon_folder_path: str,
    case_sensitive: Optional[bool] = True
) -> None:

    """
    Remove the given words from the pybear lexicon.


    Parameter
    ---------
    WORDS:
        Union[str, Sequence[str]] - the word or words to remove from
        the pybear lexicon. Cannot be an empty string or an empty
        sequence.
    lexicon_folder_path:
        str - the path to the directory that holds the lexicon text
        files.
    case_sensitive:
        Optional[bool], default = True - If True, search for the exact
        string in the fitted data. If False, normalize both the given
        string and the strings fitted on the TextStatistics instance,
        then perform the search.


    Return
    ------
    -
        None

    """



    _validate_word_input(WORDS)

    if not isinstance(lexicon_folder_path, str):
        raise TypeError(f"'lexicon_folder_path' must be a string")

    if not isinstance(case_sensitive, bool):
        raise TypeError(f"'case_sensitive' must be boolean")

    # END validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    # pizza finish

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

        MASK = np.ones(len(OLD_SUB_LEXICON), dtype=bool)
        for _word in PERTINENT_WORDS:
            if case_sensitive:
                MASK -= (_word == OLD_SUB_LEXICON)
            elif not case_sensitive:
                MASK -= (_word.upper() == np.char.upper(OLD_SUB_LEXICON))

        NEW_LEXICON = list(map(str, OLD_SUB_LEXICON[(MASK == 1)]))

        with open(full_path, 'w') as f:
            for line in NEW_LEXICON:
                f.write(line+f'\n')
            f.close()


    print(f'\n*** Lexicon update successful. ***\n')






