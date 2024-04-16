import numpy as np
from data_validation import validate_user_input as vui
from general_text import alphanumeric_str as ans, Lexicon as lex
NEW_WORDS = [
    # ********PASTE NEW WORDS HERE ******
    "WOODBRIDGE"
]
# from general_text.lexicon import lexicon_y as ly, lexicon_t as lt, lexicon_sn_sz as lsn, lexicon_z as lz, \





#     lexicon_u as lu, lexicon_x as lx, lexicon_w as lw, lexicon_v as lv, lexicon_sa_sm as lsa, lexicon_n as ln, \
#     lexicon_e as le, lexicon_r as lr, lexicon_d as ld, lexicon_f as lf, lexicon_m as lm, lexicon_a as la, \
#     lexicon_p as lp, lexicon_l as ll, lexicon_c as lc, lexicon_j as lj, lexicon_h as lh, lexicon_b as lb, \
#     lexicon_i as li, lexicon_o as lo, lexicon_g as lg, lexicon_k as lk, lexicon_q as lq

# MODULE FOR APPENDING NEW WORDS TO A SUB-LEXICON


active_letter = vui.validate_user_str(f'Enter letter of lexicon to append to > ', ans.alphabet_str_upper())

base_path = r'C:/Users/Bill/AppData/Local/Programs/Python/Python39/PyCharmProjects/general_text/lexicon/'
# base_path = r''
file_base = f'lexicon_'

raw_text = open(base_path + file_base + active_letter.lower() + '.txt', encoding="utf8").read()
OLD_LEXICON = np.char.split(np.char.replace(raw_text, f'\n', f' ')).reshape((1,-1))[0][0]
del raw_text

NEW_LEXICON = np.hstack((OLD_LEXICON, NEW_WORDS))

NEW_LEXICON = np.unique(NEW_LEXICON)   # MUST USE uniques TO TAKE OUT ANY NEW WORDS ALREADY IN LEXICON (AND SORT)

[print(_) for _ in NEW_LEXICON]

lex.Lexicon().statistics()

print(f'\nDUPLICATES:')
lex.Lexicon().find_duplicates()

print(f'\nWORDS CONTAINING NON-ALPHA CHARACTERS:')
HOLDER = []
for word in lex.Lexicon().LEXICON:
    for char in word:
        if char.upper() not in ans.alphabet_str_upper():
            HOLDER.append(word)
if len(HOLDER) > 0: [print(' '*5 + f'{_}') for _ in HOLDER]
elif len(HOLDER) == 0: print(f'None.')
print()
del HOLDER


# base_path = r'C:/Users/Bill/Desktop/'

while True:
    full_path = base_path + file_base + active_letter.lower() + f'.txt'
    __ = vui.validate_user_str(f'Going to overwrite {file_base + active_letter.lower()}.txt with words as printed. Proceed? (y)es (a)bort > ', 'YA')
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











