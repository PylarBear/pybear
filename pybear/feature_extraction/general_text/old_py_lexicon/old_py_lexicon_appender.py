import numpy as np
from data_validation import validate_user_input as vui
from general_text import alphanumeric_str as ans, Lexicon as lex
from general_text.old_py_lexicon import lexicon_y as ly, lexicon_t as lt, lexicon_sn_sz as lsn, lexicon_z as lz, \
    lexicon_u as lu, lexicon_x as lx, lexicon_w as lw, lexicon_v as lv, lexicon_sa_sm as lsa, lexicon_n as ln, \
    lexicon_e as le, lexicon_r as lr, lexicon_d as ld, lexicon_f as lf, lexicon_m as lm, lexicon_a as la, \
    lexicon_p as lp, lexicon_l as ll, lexicon_c as lc, lexicon_j as lj, lexicon_h as lh, lexicon_b as lb, \
    lexicon_i as li, lexicon_o as lo, lexicon_g as lg, lexicon_k as lk, lexicon_q as lq

# MODULE FOR APPENDING NEW WORDS TO A SUB-LEXICON

NEW_WORDS = [
    # ********PASTE NEW WORDS HERE ******

]

active_letter = vui.validate_user_str(f'Enter letter of lexicon to append to > ', ans.alphabet_str_upper())

if active_letter == 'A': OLD_LEXICON = la.lexicon_a()
elif active_letter == 'B': OLD_LEXICON = lb.lexicon_b()
elif active_letter == 'C': OLD_LEXICON = lc.lexicon_c()
elif active_letter == 'D': OLD_LEXICON = ld.lexicon_d()
elif active_letter == 'E': OLD_LEXICON = le.lexicon_e()
elif active_letter == 'F': OLD_LEXICON = lf.lexicon_f()
elif active_letter == 'G': OLD_LEXICON = lg.lexicon_g()
elif active_letter == 'H': OLD_LEXICON = lh.lexicon_h()
elif active_letter == 'I': OLD_LEXICON = li.lexicon_i()
elif active_letter == 'J': OLD_LEXICON = lj.lexicon_j()
elif active_letter == 'K': OLD_LEXICON = lk.lexicon_k()
elif active_letter == 'L': OLD_LEXICON = ll.lexicon_l()
elif active_letter == 'M': OLD_LEXICON = lm.lexicon_m()
elif active_letter == 'N': OLD_LEXICON = ln.lexicon_n()
elif active_letter == 'O': OLD_LEXICON = lo.lexicon_o()
elif active_letter == 'P': OLD_LEXICON = lp.lexicon_p()
elif active_letter == 'Q': OLD_LEXICON = lq.lexicon_q()
elif active_letter == 'R': OLD_LEXICON = lr.lexicon_r()
elif active_letter == 'S': OLD_LEXICON = np.hstack((lsa.lexicon_sa_sm(), lsn.lexicon_sn_sz()))
elif active_letter == 'T': OLD_LEXICON = lt.lexicon_t()
elif active_letter == 'U': OLD_LEXICON = lu.lexicon_u()
elif active_letter == 'V': OLD_LEXICON = lv.lexicon_v()
elif active_letter == 'W': OLD_LEXICON = lw.lexicon_w()
elif active_letter == 'X': OLD_LEXICON = lx.lexicon_x()
elif active_letter == 'Y': OLD_LEXICON = ly.lexicon_y()
elif active_letter == 'Z': OLD_LEXICON = lz.lexicon_z()

NEW_LEXICON = np.hstack((OLD_LEXICON, np.array([NEW_WORDS], dtype='<U30').reshape((1,-1))[0]))


NEW_LEXICON = np.unique(NEW_LEXICON)
[print(f' '*20 + f'"{_}",') for _ in NEW_LEXICON]


print(f'\nLEXICON STATISTICS:')
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




















