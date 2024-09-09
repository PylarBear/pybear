# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

import inspect

from typing import Iterable
from typing_extensions import Union
import numpy as np


from pybear.data_validation.arg_kwarg_validater import arg_kwarg_validater
from pybear.data_validation import validate_user_input as vui
from ._display_lexicon_update import _display_lexicon_update
from ._view_snippet import _view_snippet
from ._word_editor import _word_editor
from ._lex_lookup_add import _lex_lookup_add
from ._lex_lookup_menu import _lex_lookup_menu


def _lex_lookup(
    pizza,
    LEXICON_ADDENDUM: Union[list[str], np.ndarray[str]],
    KNOWN_WORDS: Union[list[str], np.ndarray[str], None],
    update_lexicon: bool,
    print_notes: bool
    ) -> pizza finish:

    """
    Scan entire CLEANED_TEXT object and prompt for handling of words
    not in LEXICON.

            Parameters
            ----------
            print_notes: bool - pizza finish

            Return
            ------
            pizza finish

    """
    # PUT self.CLEANED_TEXT INTO MAJUSCULE AND list_of_list FORMAT IF NOT ALREADY

    converted = False
    if not self.is_list_of_lists:
        self.as_list_of_lists()
        converted = True

    # VALIDATION, ETC ##############################################
    fxn = inspect.stack()[0][3]

    print_notes = arg_kwarg_validater(
        print_notes,
        'print_notes',
        [True, False, None],
        self.this_module,
        fxn,
        return_if_none=False
    )

    _abort = False
    if self.update_lexicon:
        MENU_DISPLAY, menu_allowed = _lex_lookup_menu(LEX_LOOK_DICT)
        if len(self.LEXICON_ADDENDUM) != 0:
            print(f'\n*** LEXICON ADDENDUM IS NOT EMPTY ***\n')
            print(f'PREVIEW OF LEXICON ADDENDUM:')
            [print(_) for _ in self.LEXICON_ADDENDUM[:10]]
            print()
            _ = vui.validate_user_str(f'EMPTY LEXICON ADDENDUM(e), PROCEED ANYWAY(p), ABORT lex_lookup(a) > ', 'AEP')
            if _ == 'A':
                _abort = True
            elif _ == 'E':
                self.LEXICON_ADDENDUM = np.empty((1,0), dtype='<U30')[0]
            del _
    elif not self.update_lexicon:
        MENU_DISPLAY, menu_allowed = _lex_lookup_menu(LEX_LOOK_DICT)

    self.KNOWN_WORDS = lx().LEXICON.copy()  #<---- pizza this was here from pre-redo
    WORDS_TO_DELETE = np.empty(0, dtype='<U30')
    SKIP_ALWAYS = np.empty(0, dtype='<U30')
    # PIZZA 2/12/23 FOR NOW, CONVERT COMMON (LOW) NUMBERS TO STRS
    EDIT_ALL_DICT = dict((
        zip(
            tuple(map(str, range(0,21))),
            ('ZERO','ONE','TWO','THREE','FOUR','FIVE','SIX','SEVEN','EIGHT',
             'NINE','TEN','ELEVEN','TWELVE','THIRTEEN','FOURTEEN','FIFTEEN',
             'SIXTEEN','SEVENTEEN','EIGHTEEN','NINETEEN','TWENTY')
        )
    ))
    SPLIT_ALWAYS_DICT = {}

    if print_notes:
        print(f'\nRunning _lexicon cross-reference...')

    number_of_edits = 0
    total_words = sum(map(len, self.CLEANED_TEXT))
    word_counter = 0
    _ = None
    for row_idx in [range(len(self.CLEANED_TEXT)) if not _abort else []][0]:
        # GO THRU BACKWARDS BECAUSE A SPLIT OR DELETE WILL CHANGE THE ARRAY OF WORDS

        if number_of_edits in range(10,int(1e6),10):
            if vui.validate_user_str(f'Save all that hard work to file(s) or continue(c) > ', 'SC') == 'S':
                if vui.validate_user_str(f'Save to csv(c) or txt(t)? > ', 'CT') == 'C':
                    self.dump_to_csv()
                else:
                    self.dump_to_txt()
            else:
                number_of_edits += 1   # SO IT DOESNT GET HUNG UP IF number_of_edits DOESNT CHANGE AS GOING THRU ROWS

        for word_idx in range(len(self.CLEANED_TEXT[row_idx])-1, -1, -1):
            word_counter += 1
            if print_notes and word_counter % 1000 == 0:
                print(f'Running word {word_counter:,} of {total_words:,}...')

            word = self.CLEANED_TEXT[row_idx][word_idx].upper()


            if self.update_lexicon and self.auto_add:
                if word in self.KNOWN_WORDS:
                    if print_notes:
                        print(f'\n*** {word} IS ALREADY IN LEXICON ***\n')
                if word not in self.KNOWN_WORDS:
                    LEXICON_ADDENDUM, KNOWN_WORDS = _lex_lookup_add(word, LEXICON_ADDENDUM, KNOWN_WORDS)
                continue


            if self.auto_delete:
                if word in self.KNOWN_WORDS:
                    if print_notes:
                        print(f'\n*** {word} IS ALREADY IN LEXICON ***\n')

                elif word not in self.KNOWN_WORDS:
                    # LOOK IF word IS 2 KNOWN WORDS SMOOSHED TOGETHER
                    # word WILL BE DELETED NO MATTER WHAT AT THIS POINT, WHETHER IT HAS REPLACEMENTS OR NOT
                    self.CLEANED_TEXT[row_idx] = np.delete(self.CLEANED_TEXT[row_idx], word_idx, axis=0)

                    if len(word) >= 4:    # LOOK FOR REPLACEMENTS IF len(word) >= 4
                        _splitter = lambda split_word: np.insert(self.CLEANED_TEXT[row_idx], word_idx, split_word.upper(), axis=0)
                        for split_idx in range(2, len(word) - 1):
                            if word[:split_idx] in self.KNOWN_WORDS and word[split_idx:] in self.KNOWN_WORDS:
                                if print_notes:
                                    print(f'\n*** SUBSTITUTING "{word}" WITH "{word[:split_idx]}" AND "{word[split_idx:]}"\n')
                                # GO THRU word BACKWARDS TO PRESERVE ORDER
                                self.CLEANED_TEXT[row_idx] = _splitter(word[split_idx:])
                                self.CLEANED_TEXT[row_idx] = _splitter(word[:split_idx])
                                break
                        # IF GET THRU for W/O FINDING A GOOD SPLIT, REPORT DELETE
                        else:
                            if print_notes:
                                print(f'\n*** DELETING {word} ***\n')
                        del _splitter
                    else:   # IF len(word) < 4, REPORT DELETE
                        if print_notes:
                            print(f'\n*** DELETING {word} ***\n')

            elif word in self.KNOWN_WORDS:
                # IMPLICIT not self.auto_delete
                if print_notes:
                    print(f'\n*** {word} IS ALREADY IN LEXICON ***\n')
            elif word in WORDS_TO_DELETE:
                # IMPLICIT not self.auto_delete
                if print_notes:
                    print(f'\n*** DELETING {word} ***\n')
                self.CLEANED_TEXT[row_idx] = np.delete(self.CLEANED_TEXT[row_idx], word_idx, axis=0)
            elif word in SKIP_ALWAYS:
                # IMPLICIT not self.auto_delete
                continue
            elif word in EDIT_ALL_DICT:
                # IMPLICIT not self.auto_delete
                self.CLEANED_TEXT[row_idx][word_idx] = EDIT_ALL_DICT[word]
            elif word in SPLIT_ALWAYS_DICT:
                # IMPLICIT not self.auto_delete
                self.CLEANED_TEXT[row_idx] = np.delete(self.CLEANED_TEXT[row_idx], word_idx, axis=0)
                for slot_idx in range(len(SPLIT_ALWAYS_DICT[word]) - 1, -1, -1):
                    # GO THRU NEW_WORDS BACKWARDS
                    self.CLEANED_TEXT[row_idx] = \
                        np.insert(self.CLEANED_TEXT[row_idx], word_idx, SPLIT_ALWAYS_DICT[word][slot_idx].upper(), axis=0)
            else:   # IMPLICIT not self.auto_delete     word IS NOT IN KNOWN_WORDS OR ANY REPETITIVE OPERATION HOLDERS
                number_of_edits += 1
                not_in_lexicon = True
                if len(word) >= 4:
                    for split_idx in range(2,len(word)-1):
                        if word[:split_idx] in self.KNOWN_WORDS and word[split_idx:] in self.KNOWN_WORDS:
                            print(_view_snippet(self.CLEANED_TEXT[row_idx], word_idx))
                            print(f"\n*{word}* IS NOT IN LEXICON\n")
                            print(f'\n*** RECOMMEND "{word[:split_idx]}" AND "{word[split_idx:]}" ***\n')
                            if vui.validate_user_str(f'Accept? (y/n) > ', 'YN') == 'Y':
                                NEW_WORDS = [word[:split_idx], word[split_idx:]]
                                _ = 'T'
                                not_in_lexicon = False
                                break
                    # else: not_in_lexicon STAYS True IF NO GOOD SPLIT FOUND
                # elif len(word) < 4: DONT DO SPLIT TEST AND not_in_lexicon STAYS True

                if not_in_lexicon:
                    print(_view_snippet(self.CLEANED_TEXT[row_idx], word_idx))
                    print(f"\n*{word}* IS NOT IN LEXICON\n")
                    _ = vui.validate_user_str(f"{MENU_DISPLAY} > ", menu_allowed)

                if _ == 'A':
                    LEXICON_ADDENDUM, KNOWN_WORDS = _lex_lookup_add(word, LEXICON_ADDENDUM, KNOWN_WORDS)
                elif _ == 'D':
                    self.CLEANED_TEXT[row_idx] = np.delete(self.CLEANED_TEXT[row_idx], word_idx, axis=0)
                elif _ in 'EF':
                    new_word = _word_editor(
                        f'ENTER NEW WORD TO REPLACE *{word}*'
                    ).upper()
                    self.CLEANED_TEXT[row_idx][word_idx] = new_word

                    if _ == 'F':
                        EDIT_ALL_DICT[word] = new_word

                    if self.update_lexicon:
                        if new_word not in self.KNOWN_WORDS and new_word not in SKIP_ALWAYS:
                            print(f"\n*{new_word}* IS NOT IN LEXICON\n")
                            SUB_MENU, sub_allowed = _lex_lookup_menu(LEX_LOOK_DICT)
                            __ = vui.validate_user_str(f"{SUB_MENU} > ", sub_allowed)
                            if __ == 'A':
                                LEXICON_ADDENDUM, KNOWN_WORDS = _lex_lookup_add(new_word, LEXICON_ADDENDUM, KNOWN_WORDS)
                            elif __ == 'K':
                                pass
                            elif __ == 'W':
                                SKIP_ALWAYS = np.insert(SKIP_ALWAYS, len(SKIP_ALWAYS), new_word, axis=0)
                            del SUB_MENU, sub_allowed
                elif _ == 'L':
                    # DELETE CURRENT ENTRY IN CURRENT ROW
                    self.CLEANED_TEXT[row_idx] = np.delete(self.CLEANED_TEXT[row_idx], word_idx, axis=0)
                    # PUT WORD INTO WORDS_TO_DELETE
                    WORDS_TO_DELETE = np.insert(WORDS_TO_DELETE, len(WORDS_TO_DELETE), word, axis=0)
                elif _ == 'K':
                    pass
                elif _ == 'W':
                    SKIP_ALWAYS = np.insert(SKIP_ALWAYS, len(SKIP_ALWAYS), word, axis=0)
                elif _ in 'SU':
                    while True:
                        new_word_ct = vui.validate_user_int(
                            f'Enter number of ways to split  *{word.upper()}*  in  *{_view_snippet(self.CLEANED_TEXT[row_idx], word_idx)}* > ', min=1, max=30)

                        NEW_WORDS = np.empty(new_word_ct, dtype='<U30')
                        for slot_idx in range(new_word_ct):
                            NEW_WORDS[slot_idx] = \
                                _word_editor(
                                    f'Enter word for slot {slot_idx + 1} (of {new_word_ct}) replacing  *{self.CLEANED_TEXT[row_idx][word_idx]}*   in  *{_view_snippet(self.CLEANED_TEXT[row_idx], word_idx)}*'
                                ).upper()

                        if vui.validate_user_str(f'User entered *{", ".join(NEW_WORDS)}* > accept? (y/n) > ', 'YN') == 'Y':
                            if _ == 'U':
                                SPLIT_ALWAYS_DICT[word] = NEW_WORDS
                            _ = 'T'  # SEND IT TO UPDATE CLEANED_TEXT WHICH IS SHARED BY WORD RECOMMENDER
                            break
                elif _ == 'Y':
                    _display_lexicon_update(LEXICON_ADDENDUM)
                    word_idx -= 1
                elif _ == 'Z':
                    break

                # NO elif !!!
                if _ == 'T':  # UPDATE CLEANED_TEXT, WHICH IS SHARED BY SPLIT AND WORD RECOMMENDER
                    self.CLEANED_TEXT[row_idx] = np.delete(self.CLEANED_TEXT[row_idx], word_idx, axis=0)
                    for slot_idx in range(len(NEW_WORDS) - 1, -1, -1):  # GO THRU NEW_WORDS BACKWARDS
                        self.CLEANED_TEXT[row_idx] = \
                            np.insert(self.CLEANED_TEXT[row_idx], word_idx, NEW_WORDS[slot_idx].upper(), axis=0)

                    if self.update_lexicon:
                        SUB_MENU, sub_allowed = _lex_lookup_menu(LEX_LOOK_DICT)
                        for slot_idx in range(len(NEW_WORDS)):
                            if NEW_WORDS[slot_idx] not in self.KNOWN_WORDS and NEW_WORDS[slot_idx] not in SKIP_ALWAYS:
                                print(f"\n*{NEW_WORDS[slot_idx]}* IS NOT IN LEXICON\n")
                                _ = vui.validate_user_str(f"{SUB_MENU} > ", sub_allowed)
                                if _ == 'A':
                                    LEXICON_ADDENDUM, KNOWN_WORDS = _lex_lookup_add(NEW_WORDS[slot_idx], LEXICON_ADDENDUM, KNOWN_WORDS)
                                elif _ == 'K':
                                    pass
                                elif _ == 'W':
                                    SKIP_ALWAYS = np.insert(SKIP_ALWAYS, len(SKIP_ALWAYS), word, axis=0)
                        del SUB_MENU, sub_allowed
                    del NEW_WORDS

        if _ == 'Z':
            break

    del MENU_DISPLAY, menu_allowed, number_of_edits, word_counter,
    del WORDS_TO_DELETE, SKIP_ALWAYS, EDIT_ALL_DICT, SPLIT_ALWAYS_DICT

    self.KNOWN_WORDS = None

    if converted: self.as_list_of_strs()
    del converted

    print(f'\n*** LEX LOOKUP COMPLETE ***\n')

    if self.update_lexicon and not _abort:
        _display_lexicon_update(LEXICON_ADDENDUM)
    del _abort
















