# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Sequence
from typing_extensions import Union

import sys, inspect, math, warnings

import numpy as np
import pandas as pd
# PIZZA NEED PLOTLY OR MATPLOTLIB

from ..._Lexicon.Lexicon import Lexicon
from ... import (
    alphanumeric_str as ans,
    _statistics as stats
)

from .....utilities._get_module_name import get_module_name
from .....utilities._view_text_snippet import _view_text_snippet
from .....data_validation import (
    validate_user_input as vui,
    arg_kwarg_validater
)



# delete_empty_rows         Remove textless rows from data.
# remove_characters         Keep only allowed or removed disallowed characters from entire CLEANED_TEXT object.
# _strip                    Remove multiple spaces and leading and trailing spaces from all text in CLEAND_TEXT object.
# normalize                 Set all text in CLEANED_TEXT object to upper case (default) or lower case.

# view_cleaned_text         Print cleaned text to screen.
# return_row_uniques        Return a potentially ragged vector containing the unique words for each row in CLEANED_TEXT object.
# view_row_uniques          Print row uniques and optionally counts to screen.
# return_overall_uniques    Return unique words in the entire CLEANED_TEXT object.
# view_overall_uniques      Print overall uniques and optionally counts to screen.

# remove_stops              Remove stop words from the entire CLEANED_TEXT object.
# justify                   Fit text as strings or as lists to user-specified number of characters per row.
# delete_words              Delete one or more words from the entire CLEANED_TEXT object.
# substitute_words          Substitute all occurrences of one or more words throughout CLEANED_TEXT.
# as_list_of_lists          Convert CLEANED_TEXT object to a possibly ragged vector of vectors, each vector containing split text.
# as_list_of_strs           Convert CLEANED_TEXT object to a single vector of strings.
# statistics                Print statistics for CLEANED_TEXT to screen.
# word_counter              Calculate frequencies for CLEANED_TEXT and print to screen or dump to file.
# dump_to_file_wrapper      Wrapper function for dumping CLEANED_TEXT object to csv or txt.
# dump_to_csv               Dump CLEANED_TEXT object to csv.
# dump_to_txt               Dump CLEANED_TEXT object to txt.
# toggle_undo               Turn undo capability ON or OFF to save memory (only used in menu()).

# STUFF FOR LEXICON LOOKUP #########################################################################################
# lex_lookup_menu           Dynamic function for returning variable menu prompts and allowed commands.
# word_editor               Validation function for single words entered by user.
# lex_lookup_add            Append a word to the LEXICON_ADDENDUM object.
# view_text_snippet         Highlights the word of interest in a series of words.
# lex_lookup                Scan entire CLEANED_TEXT object and prompt for handling of words not in LEXICON.  <<<<<
# display_lexicon_update    Prints and returns LEXICON_ADDENDUM object for copy and paste into LEXICON.
# END STUFF FOR LEXICON LOOKUP #####################################################################################




class TextCleanerMixin:


    # pizza get rid of init, only here to stop errors
    def __init__(self):

        """
        Mixin that provides methods to TextCleaner and InteractiveTextCleaner.

        """

        self.this_module = get_module_name(str(sys.modules[__name__]))

        fxn = "__init__"

        ################################################################
        # ARG / KWARG VALIDATION #######################################

        _val_X(LIST_OF_STRINGS)

        self.CLEANED_TEXT = np.array(LIST_OF_STRINGS.copy()).astype(object)
        # np.empty((1, len(self.LIST_OF_STRINGS)), dtype=object)[0]

        self.update_lexicon = arg_kwarg_validater(
            update_lexicon,
            'update_lexicon',
            [True, False, None],
            self.this_module,
            fxn,
            return_if_none=False
        )

        self.auto_add = arg_kwarg_validater(
            auto_add,
            'auto_add',
            [True, False, None],
            self.this_module,
            fxn,
            return_if_none=False
        )

        self.auto_delete = arg_kwarg_validater(
            auto_delete,
            'auto_delete',
            [True, False, None],
            self.this_module,
            fxn,
            return_if_none=False
        )

        # if self.update_lexicon is True and self.auto_delete is True:
        # 2/7/23 auto_delete CAUSES BYPASS OF HANDLING OPTIONS FOR
        # UNKNOWN WORDS DURING lex_lookup PROCESSES.  CANNOT
        # update_lexicon IF BEING BYPASSED. update_lexicon ENABLES
        # OR DISABLES THE add to _lexicon MENU OPTION DURING
        # HANDLING OF UNKNOWN WORDS
        # self._exception(f'update_lexicon AND auto_delete CANNOT BOTH BE True SIMULTANEOUSLY', fxn=fxn)

        # 3/2/23 PIZZA MAY HAVE TO REVISIT THIS LOGIC
        if self.update_lexicon is False:
            self.auto_add = False  # USER MUST HAVE OVERRODE KWARG DEFAULT
        if self.auto_delete is True:
            self.update_lexicon = False
            self.auto_add = False
        if self.auto_add is True:
            self.update_lexicon = True

        # END ARG / KWARG VALIDATION ###################################
        ################################################################

        ################################################################
        # DECLARATIONS #################################################
        self.is_list_of_lists = False

        self.LEXICON_ADDENDUM = np.empty((1, 0), dtype='<U30')[0]
        self.KNOWN_WORDS = None

        self._lexicon = Lexicon()



        # END DECLARATIONS #############################################
        ################################################################


    def delete_empty_rows(self):
        """Remove textless rows from data."""

        self.CLEANED_TEXT = _delete_empty_rows(
            self.CLEANED_TEXT
        )



    def remove_characters(
        self,
        allowed_chars_as_str: str = ans.alphanumeric_str(),
        disallowed_chars_as_str: str = None
    ):

        # 24_06_22 see the benchmark module for this.
        # winner was map_set

        """Keep only allowed or removed disallowed characters from entire CLEANED_TEXT object."""
        fxn = inspect.stack()[0][3]

        # VALIDATION ###################################################
        if not allowed_chars_as_str is None:
            if not 'STR' in str(type(allowed_chars_as_str)).upper():
                TypeError(f'allowed_chars_as_str MUST BE GIVEN AS str', fxn)
            if not disallowed_chars_as_str is None:
                ValueError(
                    f'CANNOT ENTER BOTH allowed_chars AND disallowed_chars. ONLY ONE OR THE OTHER OR NEITHER.', fxn)
        elif allowed_chars_as_str is None and disallowed_chars_as_str is None:
            ValueError(f'MUST SPECIFY ONE OF allowed_chars AND disallowed_chars.', fxn)
        elif allowed_chars_as_str is None:
            if not 'STR' in str(type(disallowed_chars_as_str)).upper():
                TypeError(f'disallowed_chars_as_str MUST BE GIVEN AS str', fxn)
        # END VALIDATION ###############################################

        if not self.is_list_of_lists:  # MUST BE LIST OF strs
            for row_idx in range(len(self.CLEANED_TEXT)):
                # GET ALL CHARS INTO A LIST, GET UNIQUES, THEN REFORM INTO A STRING OF UNIQUES
                UNIQUES = "".join(np.unique(np.fromiter((_ for _ in str(self.CLEANED_TEXT[row_idx])), dtype='<U1')))
                for char in UNIQUES:
                    if (not allowed_chars_as_str is None and char not in allowed_chars_as_str) or \
                            (not disallowed_chars_as_str is None and char in disallowed_chars_as_str):
                        self.CLEANED_TEXT[row_idx] = str(np.char.replace(str(self.CLEANED_TEXT[row_idx]), char, ''))
            del UNIQUES

        elif self.is_list_of_lists:
            # 1/21/23 WHEN LIST OF LISTS [['str1', 'str2',...], [...], ...], remove IS CAUSING SOME SLOTS TO GO TO ''.  DELETE THEM.
            for row_idx in range(len(self.CLEANED_TEXT)):
                # JOIN ROW ENTRIES W " " INTO ONE STRING, PUT INTO AN ARRAY, GET UNIQUES, REFORM AS SINGLE STRING OF UNIQUES
                UNIQUES = "".join(
                    np.unique(np.fromiter((_ for _ in " ".join(self.CLEANED_TEXT[row_idx])), dtype='<U1')))
                for char in UNIQUES:
                    if (not allowed_chars_as_str is None and char not in allowed_chars_as_str) or \
                            (not disallowed_chars_as_str is None and char in disallowed_chars_as_str):
                        self.CLEANED_TEXT[row_idx] = np.char.replace(self.CLEANED_TEXT[row_idx], char, '')

                if not f'' in self.CLEANED_TEXT[row_idx]:
                    continue
                else:
                    self.CLEANED_TEXT[row_idx] = self.CLEANED_TEXT[row_idx][..., self.CLEANED_TEXT[row_idx] != '']

            del UNIQUES

    def _strip(self):
        """Remove multiple spaces and leading and trailing spaces from all text in CLEAND_TEXT object."""
        # DO THIS ROW-WISE (SINGLE ARRAY AT A TIME), BECAUSE np.char WILL THROW A FIT IF GIVEN A RAGGED ARRAY
        for row_idx in range(len(self.CLEANED_TEXT)):
            if self.is_list_of_lists:
                while True:
                    if f'  ' in self.CLEANED_TEXT[row_idx]:
                        self.CLEANED_TEXT[row_idx] = np.char.replace(self.CLEANED_TEXT[row_idx], f'  ', f' ')
                    else:
                        map(str.strip, self.CLEANED_TEXT[row_idx])
                        break
            elif not self.is_list_of_lists:  # MUST BE LIST OF strs
                while True:
                    if f'  ' in self.CLEANED_TEXT[row_idx]:
                        self.CLEANED_TEXT[row_idx] = str(np.char.replace(self.CLEANED_TEXT[row_idx], f'  ', f' '))
                    else:
                        self.CLEANED_TEXT[row_idx] = self.CLEANED_TEXT[row_idx].strip()
                        break

    def normalize(self, upper: bool = True):  # IF NOT upper THEN lower
        """Set all text in CLEANED_TEXT object to upper case (default) or lower case."""
        # WILL PROBABLY BE A RAGGED ARRAY AND np.char WILL THROW A FIT, SO GO ROW BY ROW
        if self.is_list_of_lists:
            for row_idx in range(len(self.CLEANED_TEXT)):
                if upper:
                    self.CLEANED_TEXT[row_idx] = np.fromiter(map(str.upper, self.CLEANED_TEXT[row_idx]), dtype='U30')
                elif not upper:
                    self.CLEANED_TEXT[row_idx] = np.fromiter(map(str.lower, self.CLEANED_TEXT[row_idx]), dtype='U30')
        elif not self.is_list_of_lists:  # LIST OF strs
            if upper:
                self.CLEANED_TEXT = np.fromiter(map(str.upper, self.CLEANED_TEXT), dtype='U100000')
            elif not upper:
                self.CLEANED_TEXT = np.fromiter(map(str.lower, self.CLEANED_TEXT), dtype='U100000')

    def view_cleaned_text(self):
        """Print cleaned text to screen."""
        print(f'\nCLEANED TEXT (currently in memory as {"LISTS" if self.is_list_of_lists else "STRINGS"}):')
        [print(_) for _ in self.CLEANED_TEXT]

    def return_row_uniques(self, return_counts=False):
        """
        Return a potentially ragged vector containing the unique words
        for each row in CLEANED_TEXT object.



        """

        # MAKE BE LIST OF LISTS, THEN USE np.unique() ON EACH ROW TO FILL UNIQUES_HOLDER

        converted = False
        if not self.is_list_of_lists:
            self.as_list_of_lists()
            converted = True

        if not return_counts:

            UNIQUES = np.fromiter(map(np.unique, self.CLEANED_TEXT), dtype=object)

            # CHANGE BACK TO LIST OF strs
            if converted:
                self.as_list_of_strs()
            del converted

            return UNIQUES

        elif return_counts:
            UNIQUES_HOLDER = np.empty(len(self.CLEANED_TEXT), dtype=object)
            COUNTS_HOLDER = np.empty(len(self.CLEANED_TEXT), dtype=object)
            for row_idx in range(len(self.CLEANED_TEXT)):
                UNIQUES_HOLDER[row_idx], COUNTS_HOLDER[row_idx] = \
                    np.unique(self.CLEANED_TEXT[row_idx], return_counts=True)

            if converted: self.as_list_of_strs()
            del converted

            return UNIQUES_HOLDER, COUNTS_HOLDER

    def view_row_uniques(self, return_counts=None):
        """Print row uniques and optionally counts to screen."""

        fxn = inspect.stack()[0][3]

        return_counts = arg_kwarg_validater(
            return_counts,
            'return_counts',
            [True, False, None],
            self.this_module,
            fxn
        )

        if return_counts is None:
            return_counts = {'Y': True, 'N': False}[vui.validate_user_str(f'View counts? (y/n) > ', 'YN')]

        if return_counts is True:
            UNIQUES, COUNTS = self.return_row_uniques(return_counts=True)
        elif return_counts is False:
            UNIQUES = self.return_row_uniques(return_counts=False)

        for row_idx in range(len(UNIQUES)):
            print(f'ROW {row_idx + 1}:')
            if return_counts is True:
                for word_idx in range(len(UNIQUES[row_idx])):
                    print(f'   {UNIQUES[row_idx][word_idx]}'.ljust(30) + f'{COUNTS[row_idx][word_idx]}')
            elif return_counts is False:
                for word_idx in range(len(UNIQUES[row_idx])):
                    print(f'   {UNIQUES[row_idx][word_idx]}')
            print()

        if return_counts is True:
            del UNIQUES, COUNTS
        elif return_counts is False:
            del UNIQUES

    def return_overall_uniques(self, return_counts: bool = False):

        """Return unique words in the entire CLEANED_TEXT object."""

        if not return_counts:
            # CANT DO unique IN ONE SHOT ON self.CLEANED_TEXT BECAUSE IS LIKELY RAGGED
            return np.unique(np.hstack(self.return_row_uniques(return_counts=False)))

        elif return_counts:
            converted = False
            if not self.is_list_of_lists:
                self.as_list_of_lists()
                converted = True

            # DEFAULT IS ASCENDING
            UNIQUES, COUNTS = np.unique(np.hstack(self.CLEANED_TEXT), return_counts=True)

            if converted: self.as_list_of_strs()
            del converted

            return UNIQUES, COUNTS

    def view_overall_uniques(self, return_counts: bool = None):

        """Print overall uniques and optionally counts to screen."""

        fxn = inspect.stack()[0][3]

        return_counts = arg_kwarg_validater(
            return_counts,
            'return_counts',
            [True, False, None],
            self.this_module,
            fxn
        )

        if return_counts is None:
            return_counts = {'Y': True, 'N': False}[vui.validate_user_str(f'View counts? (y/n) > ', 'YN')]

        if return_counts is True:
            UNIQUES, COUNTS = self.return_overall_uniques(return_counts=True)

            MASK = np.flip(np.argsort(COUNTS))
            UNIQUES = UNIQUES[..., MASK]
            COUNTS = COUNTS[..., MASK]
            del MASK

            print(f'OVERALL UNIQUES:')
            [print(f'   {UNIQUES[idx]}'.ljust(30) + f'{COUNTS[idx]}') for idx in range(len(UNIQUES))]
            del UNIQUES, COUNTS

        elif return_counts is False:
            UNIQUES = self.return_overall_uniques(return_counts=False)
            print(f'OVERALL UNIQUES:')
            [print(f'   {_}') for _ in UNIQUES]
            del UNIQUES

    def remove_stops(self):
        """Remove stop words from the entire CLEANED_TEXT object."""
        converted = False
        if not self.is_list_of_lists:
            self.as_list_of_lists()
            converted = True

        for row_idx in range(len(self.CLEANED_TEXT)):
            for word_idx in range(len(self.CLEANED_TEXT[row_idx]) - 1, -1, -1):
                if self.CLEANED_TEXT[row_idx][word_idx] in self._lexicon.stop_words():
                    self.CLEANED_TEXT[row_idx] = np.delete(self.CLEANED_TEXT[row_idx], word_idx, axis=0)
                if len(self.CLEANED_TEXT) == 0:
                    break

        if converted: self.as_list_of_strs()
        del converted

    def justify(self, chars: int = None):
        """
        Fit text as strings or as lists to user-specified number of
        characters per row.

        Parameters
        ----------
        chars: int - number of characters per row


        """

        # ALSO SEE text.notepad_justifier FOR SIMILAR CODE, IF EVER CONSOLIDATING

        fxn = inspect.stack()[0][3]

        if not chars is None:
            chars = arg_kwarg_validater(
                chars,
                'characters',
                list(range(30, 50001)),
                self.this_module,
                fxn
            )
        elif chars is None:
            # DONT PUT THIS IN akv(return_if_none=)... PROMPTS USER FOR
            # INPUT BEFORE ENDING args/kwargs TO akv
            chars = vui.validate_user_int(
                f'\nEnter number of characters per line (min=30, max=50000) > ', min=30, max=50000)

        # CONVERT TO LIST OF LISTS
        converted = False
        if not self.is_list_of_lists:
            self.as_list_of_lists()
            converted = True

        seed = f''
        max_line_len = chars
        del chars
        NEW_TXT = np.empty((1, 0), dtype=f'<U{max_line_len}')[0]
        for row_idx in range(len(self.CLEANED_TEXT)):
            for word_idx in range(len(self.CLEANED_TEXT[row_idx])):
                new_word = self.CLEANED_TEXT[row_idx][word_idx]
                if len(seed) + len(new_word) <= max_line_len:
                    seed += new_word + ' '
                elif len(seed) + len(new_word) > max_line_len:
                    NEW_TXT = np.insert(NEW_TXT, len(NEW_TXT), seed.strip(), axis=0)
                    seed = new_word + ' '
        if len(seed) > 0:
            NEW_TXT = np.insert(NEW_TXT, len(NEW_TXT), seed.strip(), axis=0)

        del max_line_len, seed, new_word

        self.CLEANED_TEXT = NEW_TXT
        del NEW_TXT
        self.is_list_of_lists = False

        # OBJECT WAS WORKED ON AS LIST OF LISTS, BUT OUTPUT IS LIST OF STRS
        if converted:
            pass  # MEANING THAT IF WAS list_of_strs TO START WITH, JUST LEAVE AS IS
        elif not converted:  # OTHERWISE WAS LIST OF LISTS TO START, SO CONVERT BACK TO LIST OF LISTS
            self.as_list_of_lists()
            map(str.strip, self.CLEANED_TEXT)
        del converted

    def delete_words(self):

        """Delete one or more words from the entire CLEANED_TEXT object."""

        converted = False
        if not self.is_list_of_lists:
            self.as_list_of_lists()
            converted = True

        TO_DELETE = np.empty((1, 0), dtype='<U30')[0]

        ctr = 0
        while True:
            while True:
                to_delete = input(f'\nEnter word to delete '
                                  f'({[f"type *d* when done, " if ctr > 0 else ""][0]}*z* to abort) > ').upper()
                if to_delete in 'DZ':
                    break
                if vui.validate_user_str(f'User entered *{to_delete}* --- Accept? (y/n) > ', 'YN') == 'Y':
                    ctr += 1
                    TO_DELETE = np.insert(TO_DELETE, len(TO_DELETE), to_delete, axis=0)
            if to_delete in 'Z':
                del ctr
                break
            print(f'\nUser entered to delete')
            print(", ".join(TO_DELETE))
            if vui.validate_user_str(f'\nAccept? (y/n) > ', 'YN') == 'Y':
                del ctr
                break
            else:
                TO_DELETE = np.empty((1, 0), dtype='<U30')[0]
                ctr = 0

        if to_delete != 'Z' and len(
                TO_DELETE) > 0:  # IF USER DID NOT ABORT AND THERE ARE WORDS TO DELETE, PROCEED WITH DELETE

            for row_idx in range(len(self.CLEANED_TEXT)):
                for word_idx in range(len(self.CLEANED_TEXT[row_idx]) - 1, -1, -1):
                    if self.CLEANED_TEXT[row_idx][word_idx] in TO_DELETE:
                        self.CLEANED_TEXT[row_idx] = np.delete(self.CLEANED_TEXT[row_idx], word_idx, axis=0)

                    if len(self.CLEANED_TEXT[row_idx]) == 0: break

        del TO_DELETE, to_delete

        if converted: self.as_list_of_strs()
        del converted

    def substitute_words(self):
        """Substitute all occurrences of one or more words throughout CLEANED_TEXT."""

        converted = False
        if not self.is_list_of_lists:
            self.as_list_of_lists()
            converted = True

        TO_SUB_DICT = {}

        ctr = 0
        while True:
            replaced, replacement = '', ''
            while True:
                replaced = input(f'\nEnter word to replace '
                                 f'({["type *d* when done, " if ctr > 0 else ""][0]}*z* to abort) > ').upper()
                if replaced in 'DZ':
                    break
                else:
                    replacement = input(f'Enter word to substitute in '
                                        f'({["type *d* when done, " if ctr > 0 else ""][0]}*z* to abort) > ').upper()
                    if replacement in 'DZ': break
                if vui.validate_user_str(
                        f'User entered to replace *{replaced}* with *{replacement}*--- Accept? (y/n) > ',
                        'YN') == 'Y':
                    ctr += 1
                    TO_SUB_DICT[replaced] = replacement
            if replaced == 'Z' or replacement == 'Z': del ctr; break
            print(f'\nUser entered to replace')
            [print(f'{k} with {v}') for k, v in TO_SUB_DICT.items()]
            if vui.validate_user_str(f'\nAccept? (y/n) > ', 'YN') == 'Y': del ctr; break

        # IF USER DID NOT ABORT AND THERE ARE WORDS TO DELETE, PROCEED WITH DELETE
        if (replaced != 'Z' and replacement != 'Z') and len(TO_SUB_DICT) > 0:

            for row_idx in range(len(self.CLEANED_TEXT)):
                for word_idx in range(len(self.CLEANED_TEXT[row_idx]) - 1, -1, -1):
                    word = self.CLEANED_TEXT[row_idx][word_idx]
                    if word in TO_SUB_DICT:
                        self.CLEANED_TEXT[row_idx][word_idx] = TO_SUB_DICT[word]
            del word

        del TO_SUB_DICT, replaced, replacement

        if converted:
            self.as_list_of_strs()
        del converted


    def statistics(self):
        """Print statistics for CLEANED_TEXT to screen."""

        converted = False
        if not self.is_list_of_lists:
            self.as_list_of_lists()
            converted = True

        stats._statistics(np.hstack(self.CLEANED_TEXT))

        if converted:
            self.as_list_of_strs()
        del converted

    def word_counter(self):

        is_asc = True
        SUB_MENU = {
            'e': 'exit', 'c': 'set count cutoff', 'd': 'dump to file',
            'p': 'print table to screen', 'r': 'print chart to screen',
            's': f'change sort (currently XXX)'
        }  # DONT FORGET 's' BELOW

        cutoff_ct = 100  # SEED
        allowed = "".join(list(SUB_MENU.keys()))
        MASTER_UNIQUES, MASTER_COUNTS = self.return_overall_uniques(return_counts=True)  # DEFAULTS TO ASCENDING
        MASK = np.argsort(MASTER_COUNTS)
        MASTER_UNIQUES = MASTER_UNIQUES[..., MASK]
        MASTER_COUNTS = MASTER_COUNTS[..., MASK]
        del MASK
        WIP_UNIQUES = MASTER_UNIQUES[MASTER_COUNTS > cutoff_ct]
        WIP_COUNTS = MASTER_COUNTS[MASTER_COUNTS > cutoff_ct]

        while True:
            # MUST BE UNDER while TO RECALC is_asc
            SUB_MENU['s'] = f'change sort (currently {["ASCENDING" if is_asc else "DESCENDING"][0]})'
            display = f""
            for idx, key in enumerate(allowed):
                display += f"{SUB_MENU[key]}({key.lower()})".ljust(40)
                if idx % 3 == 2:
                    display += f"\n"

            print(display)
            selection = vui.validate_user_str(f'\nSelect operation > ', allowed).lower()

            if selection == 'e':  # 'exit'
                break
            elif selection == 'c':  # 'set count cutoff'
                cutoff_ct = vui.validate_user_int(f'\nSet count cutoff (currently {cutoff_ct}) > ', min=1)
                MASK = np.where(MASTER_COUNTS >= cutoff_ct, True, False)
                WIP_UNIQUES = MASTER_UNIQUES[MASK]
                WIP_COUNTS = MASTER_COUNTS[MASK]
                del MASK
            elif selection == 'd':  # 'dump to file'
                print(f'\nSaving WORD COUNTS to csv...')
                _core_fxn = pd.DataFrame(
                    data=np.vstack((WIP_UNIQUES, WIP_COUNTS)).transpose(),
                    columns=[f'WORD', f'COUNT']
                ).to_csv
                self.dump_to_file_wrapper(
                    _core_fxn,
                    f'.csv',
                    {'header': True, 'index': False}
                )
                print(f'\n*** Dump to csv successful. ***\n')
            elif selection == 'p':  # 'print table to screen'
                _pad = lambda x: f' ' * 5 + str(x)
                __ = 10
                print(_pad(f'WORD').ljust(2 * __) + f'FREQUENCY')
                [print(f'{_pad(WIP_UNIQUES[i])}'.ljust(2 * __) + f'{WIP_COUNTS[i]}') for i in range(len(WIP_UNIQUES))]
                print()
            elif selection == 'r':  # 'print chart to screen'
                # PIZZA CHANGE THIS TO PD.PLOT(KIND=BAR)
                df = pd.DataFrame(
                    data=WIP_COUNTS,
                    columns=['COUNT'],
                    index=WIP_UNIQUES
                )
                df.plot(kind='barh')
                # fig = px.bar(x=WIP_UNIQUES, y=WIP_COUNTS, labels={'x':'WORD','y':'COUNT'})
                # fig_widget = go.FigureWidget(fig)
                # MUST END IN .html SO IT KNOWS TO OPEN IN BROWSER
                # fig_widget.write_html('word_frequency.html', auto_open=True)
                # del fig, fig_widget
            elif selection == 's':  # 'change sort (currently XXX)'
                if is_asc:
                    is_asc = False
                elif not is_asc:
                    is_asc = True
                WIP_UNIQUES = np.flip(WIP_UNIQUES)
                MASTER_UNIQUES = np.flip(MASTER_UNIQUES)
                WIP_COUNTS = np.flip(WIP_COUNTS)
                MASTER_COUNTS = np.flip(MASTER_COUNTS)

        del SUB_MENU, display, selection, cutoff_ct, is_asc, MASTER_UNIQUES, MASTER_COUNTS
        try:
            del WIP_UNIQUES
        except:
            pass
        try:
            del WIP_COUNTS
        except:
            pass





    ####################################################################
    ####################################################################
    # STUFF FOR LEXICON LOOKUP #########################################



    def word_editor(self, word, prompt=None):
        """Validation function for single words entered by user."""

        # pizza only used in lex_lookup

        while True:
            word = input(f'{prompt} > ').upper()
            if vui.validate_user_str(f'USER ENTERED *{word}* -- ACCEPT? (Y/N) > ', 'YN') == 'Y':
                break

        return word

    def lex_lookup_add(self, word):
        """Append a word to the LEXICON_ADDENDUM object."""

        # pizza only used in lex_lookup

        self.LEXICON_ADDENDUM = \
            np.insert(self.LEXICON_ADDENDUM, len(self.LEXICON_ADDENDUM), word, axis=0)
        self.KNOWN_WORDS = \
            np.insert(self.KNOWN_WORDS, len(self.KNOWN_WORDS), word, axis=0)


    def _view_snippet(self, VECTOR, idx, span=9):

        """
        Highlights the word of interest (which is given by the 'idx'
        value) in a series of words. For example, in a simple case that
        avoids edge effects, a span of 9 would show 4 strings to the
        left of the target string in lower-case, the target string itself
        capitalized, then the 4 strings to the right of the target string
        in lower-case.


        Parameters
        ----------
        VECTOR:
            Sequence[str] - the sequence of strings that provides a
            subsequence of strings to highlight. Cannot be empty.
        idx:
            numbers.Integral - the index of the string in the sequence
            of strings to highlight.
        span:
            Optional[numbers.Integral], default = 9 - the number of
            strings in the sequence of strings to select when
            highlighting one particular central string.


        Return
        ------
        -
            str: the highlighted portion of the string sequence.


        """

        _view_text_snippet(VECTOR, idx, span=span)


    def lex_lookup(self, print_notes: bool = False):

        """
        Scan entire CLEANED_TEXT object and prompt for handling of words
        not in LEXICON.



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
            MENU_DISPLAY, menu_allowed = self.lex_lookup_menu(fxn=fxn)
            if len(self.LEXICON_ADDENDUM) != 0:
                print(f'\n*** LEXICON ADDENDUM IS NOT EMPTY ***\n')
                print(f'PREVIEW OF LEXICON ADDENDUM:')
                [print(_) for _ in self.LEXICON_ADDENDUM[:10]]
                print()
                _ = vui.validate_user_str(f'EMPTY LEXICON ADDENDUM(e), PROCEED ANYWAY(p), ABORT lex_lookup(a) > ',
                                          'AEP')
                if _ == 'A':
                    _abort = True
                elif _ == 'E':
                    self.LEXICON_ADDENDUM = np.empty((1, 0), dtype='<U30')[0]
                del _
        elif not self.update_lexicon:
            MENU_DISPLAY, menu_allowed = self.lex_lookup_menu(allowed='desk', fxn=fxn)

        self.KNOWN_WORDS = self._lexicon.lexicon_.copy()
        WORDS_TO_DELETE = np.empty(0, dtype='<U30')
        SKIP_ALWAYS = np.empty(0, dtype='<U30')
        # PIZZA 2/12/23 FOR NOW, CONVERT COMMON (LOW) NUMBERS TO STRS
        EDIT_ALL_DICT = dict((
            zip(
                tuple(map(str, range(0, 21))),
                ('ZERO', 'ONE', 'TWO', 'THREE', 'FOUR', 'FIVE', 'SIX', 'SEVEN', 'EIGHT',
                 'NINE', 'TEN', 'ELEVEN', 'TWELVE', 'THIRTEEN', 'FOURTEEN', 'FIFTEEN',
                 'SIXTEEN', 'SEVENTEEN', 'EIGHTEEN', 'NINETEEN', 'TWENTY')
            )
        ))
        SPLIT_ALWAYS_DICT = {}

        if print_notes: print(f'\nRunning _lexicon cross-reference...')

        number_of_edits = 0
        total_words = sum(map(len, self.CLEANED_TEXT))
        word_counter = 0
        _ = None
        for row_idx in [range(len(self.CLEANED_TEXT)) if not _abort else []][0]:
            # GO THRU BACKWARDS BECAUSE A SPLIT OR DELETE WILL CHANGE THE ARRAY OF WORDS

            if number_of_edits in range(10, int(1e6), 10):
                if vui.validate_user_str(f'Save all that hard work to file(s) or continue(c) > ', 'SC') == 'S':
                    if vui.validate_user_str(f'Save to csv(c) or txt(t)? > ', 'CT') == 'C':
                        self.dump_to_csv()
                    else:
                        self.dump_to_txt()
                else:
                    number_of_edits += 1  # SO IT DOESNT GET HUNG UP IF number_of_edits DOESNT CHANGE AS GOING THRU ROWS

            for word_idx in range(len(self.CLEANED_TEXT[row_idx]) - 1, -1, -1):
                word_counter += 1
                if print_notes and word_counter % 1000 == 0:
                    print(f'Running word {word_counter:,} of {total_words:,}...')

                word = self.CLEANED_TEXT[row_idx][word_idx].upper()

                if self.update_lexicon and self.auto_add:
                    if word in self.KNOWN_WORDS:
                        if print_notes: print(f'\n*** {word} IS ALREADY IN LEXICON ***\n')
                    if word not in self.KNOWN_WORDS: self.lex_lookup_add(word)
                    continue

                if self.auto_delete:
                    if word in self.KNOWN_WORDS:
                        if print_notes:
                            print(f'\n*** {word} IS ALREADY IN LEXICON ***\n')

                    elif word not in self.KNOWN_WORDS:
                        # LOOK IF word IS 2 KNOWN WORDS SMOOSHED TOGETHER
                        # word WILL BE DELETED NO MATTER WHAT AT THIS POINT, WHETHER IT HAS REPLACEMENTS OR NOT
                        self.CLEANED_TEXT[row_idx] = np.delete(self.CLEANED_TEXT[row_idx], word_idx, axis=0)

                        if len(word) >= 4:  # LOOK FOR REPLACEMENTS IF len(word) >= 4
                            _splitter = lambda split_word: np.insert(self.CLEANED_TEXT[row_idx], word_idx,
                                                                     split_word.upper(), axis=0)
                            for split_idx in range(2, len(word) - 1):
                                if word[:split_idx] in self.KNOWN_WORDS and word[split_idx:] in self.KNOWN_WORDS:
                                    if print_notes:
                                        print(
                                            f'\n*** SUBSTITUTING "{word}" WITH "{word[:split_idx]}" AND "{word[split_idx:]}"\n')
                                    # GO THRU word BACKWARDS TO PRESERVE ORDER
                                    self.CLEANED_TEXT[row_idx] = _splitter(word[split_idx:])
                                    self.CLEANED_TEXT[row_idx] = _splitter(word[:split_idx])
                                    break
                            # IF GET THRU for W/O FINDING A GOOD SPLIT, REPORT DELETE
                            else:
                                if print_notes:
                                    print(f'\n*** DELETING {word} ***\n')
                            del _splitter
                        else:  # IF len(word) < 4, REPORT DELETE
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
                            np.insert(self.CLEANED_TEXT[row_idx], word_idx, SPLIT_ALWAYS_DICT[word][slot_idx].upper(),
                                      axis=0)
                else:  # IMPLICIT not self.auto_delete     word IS NOT IN KNOWN_WORDS OR ANY REPETITIVE OPERATION HOLDERS
                    number_of_edits += 1
                    not_in_lexicon = True
                    if len(word) >= 4:
                        for split_idx in range(2, len(word) - 1):
                            if word[:split_idx] in self.KNOWN_WORDS and word[split_idx:] in self.KNOWN_WORDS:
                                print(self._view_snippet(self.CLEANED_TEXT[row_idx], word_idx))
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
                        print(self._view_snippet(self.CLEANED_TEXT[row_idx], word_idx))
                        print(f"\n*{word}* IS NOT IN LEXICON\n")
                        _ = vui.validate_user_str(f"{MENU_DISPLAY} > ", menu_allowed)

                    if _ == 'A':
                        self.lex_lookup_add(word)
                    elif _ == 'D':
                        self.CLEANED_TEXT[row_idx] = np.delete(self.CLEANED_TEXT[row_idx], word_idx, axis=0)
                    elif _ in 'EF':
                        new_word = self.word_editor(word, prompt=f'ENTER NEW WORD TO REPLACE *{word}*').upper()
                        self.CLEANED_TEXT[row_idx][word_idx] = new_word

                        if _ == 'F':
                            EDIT_ALL_DICT[word] = new_word

                        if self.update_lexicon:
                            if new_word not in self.KNOWN_WORDS and new_word not in SKIP_ALWAYS:
                                print(f"\n*{new_word}* IS NOT IN LEXICON\n")
                                SUB_MENU, sub_allowed = self.lex_lookup_menu(allowed='AKW')
                                __ = vui.validate_user_str(f"{SUB_MENU} > ", sub_allowed)
                                if __ == 'A':
                                    self.lex_lookup_add(new_word)
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
                                f'Enter number of ways to split  *{word.upper()}*  in  *{self._view_snippet(self.CLEANED_TEXT[row_idx], word_idx)}* > ',
                                min=1, max=30)

                            NEW_WORDS = np.empty(new_word_ct, dtype='<U30')
                            for slot_idx in range(new_word_ct):
                                NEW_WORDS[slot_idx] = \
                                    self.word_editor(word.upper(),
                                                     prompt=f'Enter word for slot {slot_idx + 1} (of {new_word_ct}) replacing  *{self.CLEANED_TEXT[row_idx][word_idx]}*  '
                                                            f'in  *{self._view_snippet(self.CLEANED_TEXT[row_idx], word_idx)}*'
                                                     ).upper()

                            if vui.validate_user_str(f'User entered *{", ".join(NEW_WORDS)}* > accept? (y/n) > ',
                                                     'YN') == 'Y':
                                if _ == 'U':
                                    SPLIT_ALWAYS_DICT[word] = NEW_WORDS
                                _ = 'T'  # SEND IT TO UPDATE CLEANED_TEXT WHICH IS SHARED BY WORD RECOMMENDER
                                break
                    elif _ == 'Y':
                        self.display_lexicon_update()
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
                            SUB_MENU, sub_allowed = self.lex_lookup_menu(allowed='AKW')
                            for slot_idx in range(len(NEW_WORDS)):
                                if NEW_WORDS[slot_idx] not in self.KNOWN_WORDS and NEW_WORDS[
                                    slot_idx] not in SKIP_ALWAYS:
                                    print(f"\n*{NEW_WORDS[slot_idx]}* IS NOT IN LEXICON\n")
                                    _ = vui.validate_user_str(f"{SUB_MENU} > ", sub_allowed)
                                    if _ == 'A':
                                        self.lex_lookup_add(NEW_WORDS[slot_idx])
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

        if self.update_lexicon and not _abort: self.display_lexicon_update()
        del _abort

    def display_lexicon_update(self):
        """Prints and returns LEXICON_ADDENDUM object for copy and paste into LEXICON."""
        if len(self.LEXICON_ADDENDUM) != 0:
            self.LEXICON_ADDENDUM.sort()
            print(f'\n *** COPY AND PASTE THESE WORDS INTO LEXICON:\n')
            print(f'[')
            [print(f'   "{_}"{"" if _ == self.LEXICON_ADDENDUM[-1] else ","}') for _ in self.LEXICON_ADDENDUM]
            print(f']')
            print()

            _ = f'dum' + input(f'\n*** Paused to allow copy, hit Enter to continue > ');
            del _

            return self.LEXICON_ADDENDUM
        else:
            print(f'\n *** LEXICON ADDENDUM IS EMPTY *** \n')

    # END STUFF FOR LEXICON LOOKUP #####################################
    ####################################################################
    ####################################################################










































