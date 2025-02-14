# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



# pizza this probably should go when _add_word is done uce!



import os

import numpy as np

from pybear.data_validation import validate_user_input as vui
from pybear.feature_extraction.text import alphanumeric_str as ans
from pybear.feature_extraction.text._Lexicon import Lexicon as lex



NEW_WORDS = [
    # ********PASTE NEW WORDS HERE ******
]





# MODULE FOR APPENDING NEW WORDS TO A SUB-LEXICON


active_letter = vui.validate_user_str(
    f'Enter letter of lexicon to append to > ',
    ans.alphabet_str_upper()
)

module_dir = os.path.dirname(os.path.abspath(__file__))
file_base = f'lexicon_'
full_path = os.path.join(module_dir, file_base, active_letter.lower(), '.txt')
del module_dir, file_base

with open(full_path, 'r') as f:
    raw_text = np.fromiter(f, dtype='<U40')

OLD_LEXICON = np.char.replace(raw_text, f'\n', f' ')
del raw_text

NEW_LEXICON = np.hstack((OLD_LEXICON, NEW_WORDS))

# MUST USE uniques TO TAKE OUT ANY NEW WORDS ALREADY IN LEXICON (AND SORT)
NEW_LEXICON = np.unique(NEW_LEXICON)

[print(_) for _ in NEW_LEXICON]

LEX = lex.Lexicon()

LEX.statistics()

print(f'\nDUPLICATES:')
LEX.find_duplicates()

print(f'\nWORDS CONTAINING NON-ALPHA CHARACTERS:')
HOLDER = []
for word in LEX.LEXICON:
    for char in word:
        if char.upper() not in ans.alphabet_str_upper():
            HOLDER.append(word)
if len(HOLDER) > 0:
    [print(' '*5 + f'{_}') for _ in HOLDER]
elif len(HOLDER) == 0:
    print(f'None.')
print()

del HOLDER



while True:

    __ = vui.validate_user_str(
        f'Going to overwrite {full_path} with words as printed. '
        f'Proceed? (y)es (a)bort > ',
        'YA'
    )

    if __ == 'Y':
        with open(full_path, 'w') as f:
            for line in NEW_LEXICON:
                f.write(line+f'\n')
            f.close()

        print(f'\n*** Dump to txt successful. ***\n')
        break

    elif __ == 'A':
        print(f'\n*** ABORTED BY USER. ***\n')
        break











