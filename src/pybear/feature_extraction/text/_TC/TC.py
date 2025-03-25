# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Optional, Sequence
from typing_extensions import Union
import numpy.typing as npt
from ._type_aliases import MenuDictType

import numbers
import re

import numpy as np

from ._validation._X import _val_X
from ._validation._auto_add import _val_auto_add
from ._validation._auto_delete import _val_auto_delete
from ._validation._update_lexicon import _val_update_lexicon

from .._StopRemover.StopRemover import StopRemover
from .._TextLookup.TextLookupRealTime import TextLookupRealTime
from .._TextNormalizer.TextNormalizer import TextNormalizer
from .._TextRemover.TextRemover import TextRemover
from .._TextReplacer.TextReplacer import TextReplacer
from .._TextStripper.TextStripper import TextStripper

from ._validation._menu import _menu_validation

from .._Lexicon.Lexicon import Lexicon
from .. import alphanumeric_str as ans

from ....base.mixins._FileDumpMixin import FileDumpMixin

from ....data_validation import (
    validate_user_input as vui,
    arg_kwarg_validater
)




# delete_empty_rows         Remove textless rows from data.
# remove_characters         Keep only allowed or removed disallowed characters from entire CLEANED_TEXT object.
# strip                    Remove multiple spaces and leading and trailing spaces from all text in CLEAND_TEXT object.
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
# dump_to_csv               Dump CLEANED_TEXT object to csv.
# dump_to_txt               Dump CLEANED_TEXT object to txt.
# toggle_undo               Turn undo capability ON or OFF to save memory (only used in menu()).



# pizza dont forget mixins!
class TC(FileDumpMixin):


    def __init__(
        self,
        X: Sequence[str],
        update_lexicon: Optional[bool] = False,
        auto_add: Optional[Union[bool, None]] = False,
        auto_delete: Optional[bool] = False
    ) -> None:  # return_as_list_of_lists=False,  # pizza what this mean?

        """
        Pizza


        Parameters
        ----------
        X:
            Sequence[str] -
        update_lexicon:
            bool=False -
        auto_add:
            bool=False -
        auto_delete:
            bool=False

        # pizza
        # X MUST BE PASSED AS ['str1','str2','str3'...],
        # JUST LIKE NNLM50

        # auto_add AUTOMATICALLY ADDS AN UNKNOWN WORD TO LEXICON_UPDATE
        # W/O PROMPTING USER
        # (JUST GOES ALL THE WAY THRU WITHOUT PROMPTS) AUTOMATICALLY
        # SENSES AND MAKES 2-WAY SPLITS


        Return
        ------
        -
            None

        """


        ################################################################
        # ARG / KWARG VALIDATION #######################################

        _val_X(X)

        # pizza as of 25_02_19_12_48_00 this fails when dtype is not '<U10000'
        # come back to this and see if can make this take f'<U{min(map(len, list(X)))}'
        self.X = np.fromiter(map(str, list(X)), dtype=f'<U10000')

        self.CLEANED_TEXT = self.X.copy().astype(object)

        _val_update_lexicon(auto_add)
        self.update_lexicon: bool = update_lexicon or False

        _val_auto_add(auto_add)
        self.auto_add: bool = auto_add or False

        _val_auto_delete(auto_delete)
        self.auto_delete = auto_delete or False

        # if self.update_lexicon is True and self.auto_delete is True:
            # 2/7/23 auto_delete CAUSES BYPASS OF HANDLING OPTIONS FOR
            # UNKNOWN WORDS DURING lex_lookup PROCESSES. CANNOT
            # update_lexicon IF BEING BYPASSED. update_lexicon ENABLES
            # OR DISABLES THE add to _lexicon MENU OPTION DURING
            # HANDLING OF UNKNOWN WORDS
            # raise ValueError(f'update_lexicon AND auto_delete CANNOT BOTH BE True SIMULTANEOUSLY')

        # 3/2/23 PIZZA MAY HAVE TO REVISIT THIS LOGIC
        if self.update_lexicon is False:
            self.auto_add = False   # USER MUST HAVE OVERRODE KWARG DEFAULT
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

        self.LEXICON_ADDENDUM: npt.NDArray[str] = np.empty((1,0), dtype='<U30')[0]

        self.KNOWN_WORDS = Lexicon().lexicon_.copy()
        self._stop_words = Lexicon().stop_words_.copy()

        self.LEX_LOOK_DICT = {
            'A': 'ADD TO LEXICON', 'D': 'DELETE', 'E': 'EDIT',
            'F': 'EDIT ALL', 'L': 'DELETE ALL', 'S': 'SPLIT',
            'U': 'SPLIT ALWAYS', 'K': 'SKIP ONCE', 'W': 'SKIP ALWAYS',
            'Y': 'VIEW LEXICON ADDENDUM', 'Z': 'GO TO MAIN MENU'
        }  # DONT USE 'T' !!!   pizza why?

        if not self.update_lexicon:
            del self.LEX_LOOK_DICT['A']


        self.lex_look_allowed = "".join(list(self.LEX_LOOK_DICT.keys())).lower()
        self.undo_status = False
        self.CLEANED_TEXT_BACKUP = None

        self.MENU_DICT: MenuDictType = {
            'D': {'label':'delete_empty_rows', 'function': self.delete_empty_rows},
            'R': {'label':'remove_characters',  'function': self.remove_characters},
            'S': {'label':'strip', 'function': self.strip},
            'N': {'label':'normalize', 'function': self.normalize},
            # 'A': {'label': 'statistics', 'function': self.statistics},
            # 'K': {'label': 'word_counter', 'function': self.word_counter},
            'V': {'label': 'view_CLEANED_TEXT', 'function': self.view_cleaned_text},
            # 'U': {'label': 'view_row_uniques', 'function': self.view_row_uniques},
            # 'O': {'label': 'view_overall_uniques', 'function': self.view_overall_uniques},
            # 'Y': {'label':'view_lexicon_addendum', 'function': self.display_lexicon_update},
            'T': {'label':'remove_stops', 'function': self.remove_stops},
            'J': {'label':'justify', 'function': self.justify},
            'W': {'label':'delete_words', 'function': self.delete_words},
            'B': {'label':'substitute_words', 'function': self.substitute_words},
            'L': {'label':'as_list_of_lists', 'function': self.as_list_of_lists},
            'I': {'label':'as_list_of_strs', 'function': self.as_list_of_strs},
            'P': {'label':'lex_lookup', 'function': self.lex_lookup},
            'C': {'label':'dump_to_csv', 'function': self.dump_to_csv},
            'X': {'label':'dump_to_txt', 'function': self.dump_to_txt},
            'E': {'label':f'toggle UNDO (currently {self.undo_status})', 'function': self.toggle_undo},
            'F': {'label':'undo', 'function': None},
            'Q': {'label':'quit', 'function': None},
            'Z': {'label':'accept and exit', 'function': None}
        }

        # END DECLARATIONS #############################################
        ################################################################


    def menu(
        self,
        allowed: Optional[Union[str, None]] = None,   # pizza Literal?
        disallowed: Optional[Union[str, None]] = None
    ) -> None:

        """
        Dynamic function for returning variable menu prompts and allowed
        commands. Both cannot simultaneously be strings. If both are
        simultaneously None, then all keys are allowed. Inputs are
        case-sensitive.


        Parameters
        ----------
        allowed:
            str - the keys of MENU_DICT that are allowed to be accessed.
        disallowed:
            str - the keys of MENU_DICT that are not allowed to be accessed.


        """


        _possible_keys = "".join(self.MENU_DICT.keys())

        _menu_validation(_possible_keys, allowed, disallowed)

        if disallowed is not None:  # then 'allowed' is None
            allowed = "".join([_ for _ in _possible_keys if _ not in disallowed])
        elif allowed is None and disallowed is None:
            allowed = _possible_keys


        while True:

            # BUILD MENU DISPLAY #######################################
            # MUST BE INSIDE while BECAUSE PRINTOUT IS DYNAMIC BASED ON
            # undo_status AND AVAILABILITY OF BACKUP
            # MANAGE AVAILABILITY OF undo COMMAND BASED ON STATUS OF
            # undo_status AND BACKUP
            if self.undo_status is False or (self.undo_status is True and self.CLEANED_TEXT_BACKUP is None):
                allowed = allowed.replace('F', '')
            elif self.undo_status is True and not self.CLEANED_TEXT_BACKUP is None and 'F' not in allowed:
                allowed += 'F'

            # MUST UPDATE "UNDO TOGGLE" 'label' VALUE EVERY TIME EXPLICITLY,
            # OTHERWISE IT IS NOT UPDATING.  IT APPEARS THAT THE VALUE IS
            # BEING SET THE ONE TIME DURING init, BUT ONCE WRIT THE VALUE
            # IS HARD-CODED AND NO LONGER SEES A {} THERE.
            self.MENU_DICT['E']['label'] = f'toggle UNDO (currently {self.undo_status})'

            display = f""
            for idx, key in enumerate(allowed):
                display += f"{self.MENU_DICT[key]['label']}({key.lower()})".ljust(40)
                if idx % 3 == 2:
                    display += f"\n"
            # END BUILD MENU DISPLAY ###################################
            print()
            print(display)

            selection = vui.validate_user_str(f'\nSelect operation > ', allowed)

            # ONLY CREATE A NEW BACKUP COPY WHEN GOING TO delete_empty_rows,
            # remove_characters, strip, normalize,
            # remove_stops, justify, lex_lookup, delete_words, substitute_words
            if self.undo_status is True and selection in 'DRSNTJPWB':
                self.CLEANED_TEXT_BACKUP = self.CLEANED_TEXT.copy()

            if selection == 'F':
                self.CLEANED_TEXT = self.CLEANED_TEXT_BACKUP.copy()
                self.CLEANED_TEXT_BACKUP = None
                continue    # FORCE BYPASS AROUND BELOW FUNCTION CALL
            elif selection == 'Q':
                raise ValueError(f'USER TERMINATED')
            elif selection == 'Z':
                del display
                return self.CLEANED_TEXT   # 4-8-23 ALLOW RETURN OF CLEANED_TEXT FROM menu()

            self.MENU_DICT[selection]['function']()

            print()


    def delete_empty_rows(self):
        """Remove textless rows from data."""
        if not self.is_list_of_lists:    # MUST BE list_of_strs
            for row_idx in range(len(self.CLEANED_TEXT)-1, -1, -1):
                if self.CLEANED_TEXT[row_idx] in ['', ' ', '\n', '\t']:
                    self.CLEANED_TEXT = np.delete(self.CLEANED_TEXT, row_idx, axis=0)

        if self.is_list_of_lists:
            for row_idx in range(len(self.CLEANED_TEXT) - 1, -1, -1):
                for EMPTY_OBJECT in [[''], [' '], ['\n'], ['\t']]:
                    if np.array_equiv(self.CLEANED_TEXT[row_idx], EMPTY_OBJECT):
                        self.CLEANED_TEXT = np.delete(self.CLEANED_TEXT, row_idx, axis=0)


    def remove_characters(
        self,
        allowed_chars:Optional[Union[str, None]] = ans.alphanumeric_str(),
        disallowed_chars:Optional[Union[str, None]] = None
    ) -> None:

        """
        Remove characters that are not allowed or are explicitly
        disallowed from the data. allowed_chars and disallowed_chars
        cannot simultaneously be strings and cannot simultaneously be
        None.


        Parameter
        ---------
        allowed_chars:
            str - the characters that are to be kept; cannot be passed
            if disallowed_chars is passed.
        disallowed_chars:
            str - the characters that are to be removed; cannot be passed
            if allowed_chars is passed.


        Return
        ------
        -
            None


        """


        Trfm = TextReplacer(regexp_replace=(f'[^{allowed_chars}]', ''))
        self.CLEANED_TEXT = Trfm.fit_transform(self.CLEANED_TEXT)
        del Trfm

        # can be 1D or 2D, returns same dim as given
        Trfm2 = TextRemover(str_remove='')
        self.CLEANED_TEXT = Trfm2.fit_transform(self.CLEANED_TEXT)


        # self.CLEANED_TEXT = _remove_characters(
        #     self.CLEANED_TEXT,
        #     self.is_list_of_lists,
        #     allowed_chars,
        #     disallowed_chars
        # )


    def strip(self) -> None:

        """
        Remove multiple spaces and leading and trailing spaces from all
        text in the data.


        Parameters
        ----------
        _WIP_X:
            Union[list[str], list[list[str]], npt.NDArray[str]] - The
            data object. Must be a list of strings, a list of lists of
            strings, or a numpy array of strings.
        _is_2D:
            bool - whether the data object is 1D or 2D.


        Return
        ------
        -
            Union[list[str], list[list[str]], npt.NDArray[str]] - the
            data less any unnecessary spaces.

        """

        # takes 1 or 2D. returns same dim as given
        self.CLEANED_TEXT = TextStripper().fit_transform(self.CLEANED_TEXT)
        # takes 1 or 2D. returns same dim as given
        _kwargs = {'regexp_replace': {(re.compile(f' +'), ' '), (f' ,', f',')}}
        self.CLEANED_TEXT = TextReplacer(**_kwargs).fit_transform(self.CLEANED_TEXT)


    def normalize(self, upper:Optional[bool] = True) -> None:

        """
        Set all text in the data to upper case (default) or lower case.


        Parameters
        ----------
        upper:
            Optional[bool], default=True - the case to normalize to;
            upper case if True, lower case if False.


        Return
        ------
        -
            Union[list[str], list[list[str]], npt.NDArray[str]] - the
            data with normalized text.


        """


        # can be 1 or 2D. returned with same dim.
        self.CLEANED_TEXT = TextNormalizer(upper=upper).fit_transform(self.CLEANED_TEXT)


    def view_cleaned_text(self) -> None:
        """Print cleaned text to screen."""
        print(f'\nCLEANED TEXT (currently in memory as {"LISTS" if self.is_list_of_lists else "STRINGS"}):')
        [print(_) for _ in self.CLEANED_TEXT]


    def remove_stops(self):
        """Remove stop words from the entire CLEANED_TEXT object."""

        converted = False
        if not self.is_list_of_lists:
            self.as_list_of_lists()
            converted = True


        Trfm = StopRemover(remove_empty_rows=True, n_jobs=1)
        self.CLEANED_TEXT = Trfm.fit_transform(self.CLEANED_TEXT)
        del Trfm

        if converted:
            self.as_list_of_strs()
        del converted


    def justify(
        self,
        chars:Optional[Union[numbers.Integral, None]] = None
    ) -> None:

        """
        Fit text as strings or as lists to user-specified number of
        characters per row.

        Parameters
        ----------
        chars:
            int - number of characters per row


        """


        # CONVERT TO LIST OF LISTS
        converted = False
        if not self.is_list_of_lists:
            self.as_list_of_lists()
            converted = True


        # ALSO SEE text.notepad_justifier FOR SIMILAR CODE, IF EVER CONSOLIDATING

        if not chars is None:
            arg_kwarg_validater(
                chars,
                'characters',
                list(range(30,50001)),
                'TC',
                'justify'
            )
        elif chars is None:
            # DONT PUT THIS IN akv(return_if_none=)... PROMPTS USER FOR
            # INPUT BEFORE PASSING TO akv
            chars = vui.validate_user_int(
                f'\nEnter number of characters per line (min=30, max=50000) > ', min=30, max=50000)



        seed = f''
        max_line_len = chars
        del chars
        NEW_TXT = np.empty((1,), dtype=f'<U{max_line_len}')
        for row_idx in range(len(self.CLEANED_TEXT)):
            for word_idx in range(len(self.CLEANED_TEXT[row_idx])):
                new_word = self.CLEANED_TEXT[row_idx][word_idx]
                if len(seed) + len(new_word) <= max_line_len:
                    seed += new_word+' '
                elif len(seed) + len(new_word) > max_line_len:
                    NEW_TXT = np.insert(NEW_TXT, len(NEW_TXT), seed.strip(), axis=0)
                    seed = new_word+' '
        if len(seed) > 0:
            NEW_TXT = np.insert(NEW_TXT, len(NEW_TXT), seed.strip(), axis=0)

        del max_line_len, seed, new_word

        self.CLEANED_TEXT = NEW_TXT
        del NEW_TXT
        self.is_list_of_lists = False


        # OBJECT WAS WORKED ON AS LIST OF LISTS, BUT OUTPUT IS LIST OF STRS
        if converted:
            # MEANING THAT IS WAS list_of_strs TO START WITH, JUST LEAVE AS IS
            pass
        elif not converted:
            # OTHERWISE WAS LIST OF LISTS TO START, SO CONVERT BACK TO LIST OF LISTS
            self.as_list_of_lists()
            map(str.strip, self.CLEANED_TEXT)
        del converted


    def delete_words(self):

        """Delete one or more words from the entire CLEANED_TEXT object."""

        # pizza as of 25_03_25 this isnt tested

        converted = False
        if not self.is_list_of_lists:
            self.as_list_of_lists()
            converted = True

        TO_DELETE = []

        while True:
            while True:
                to_delete = input(f'\nEnter word to delete (case-sensitive) > ')

                _prompt = f'User entered *{to_delete}* --- Accept? (y/n) > '
                if vui.validate_user_str(_prompt, 'YN') == 'Y':
                    TO_DELETE.append(to_delete)

                print(f'\nCurrent words to be deleted:')
                print(", ".join(TO_DELETE))
                if vui.validate_user_str(f'\nEnter another? > ', 'YN') == 'N':
                    break

            print(f'\nUser entered to delete')
            print(", ".join(TO_DELETE))
            _opt = vui.validate_user_str(f'\nAccept? (y/n) Abort? (a) > ', 'YNA')
            if _opt == 'Y':
                pass
            elif _opt == 'N':
                TO_DELETE = []
                continue
            elif _opt == 'A':
                break
            else:
                raise Exception

            # IF USER DID NOT ABORT AND THERE ARE WORDS TO DELETE
            if len(TO_DELETE) > 0:

                # can be 1D or 2D, returns same dim as given
                Trfm = TextRemover(str_remove=set(TO_DELETE))
                self.CLEANED_TEXT = Trfm.fit_transform(self.CLEANED_TEXT)
                del Trfm

            del TO_DELETE
            break

        if converted:
            self.as_list_of_strs()
        del converted


    def substitute_words(self):
        """
        Substitute all occurrences of one or more words throughout
        CLEANED_TEXT.

        """

        # pizza as of 25_03_25 this isnt tested

        converted = False
        if not self.is_list_of_lists:
            self.as_list_of_lists()
            converted = True


        TO_SUB_DICT = {}

        while True:
            while True:
                replaced = input(f'\nEnter word to replace (case-sensitive) > ')
                _prompt = f'User entered *{replaced}*, accept? (y/n)'
                if vui.validate_user_str(_prompt, 'YN') == 'N':
                    continue

                replacement = input(f'Enter word to substitute for *{replaced}* > ')
                _prompt = f'Replace *{replaced}* with *{replacement}*, accept? (y/n)'
                if vui.validate_user_str(_prompt, 'YN') == 'N':
                    continue

                TO_SUB_DICT[replaced] = replacement

                print(f'Current substitutions:')
                for k, v in TO_SUB_DICT.items():
                    print(f'{k} : {v}')

                if vui.validate_user_str(f'\nEnter another? (y/n) > ', 'YN') == 'N':
                    break

            print(f'Current substitutions:')
            for k, v in TO_SUB_DICT.items():
                print(f'{k} : {v}')

            _opt = vui.validate_user_str(f'\nAccept? (y/n) Abort? (a) > ', 'YNA')
            if _opt == 'Y':
                pass
            elif _opt == 'N':
                TO_SUB_DICT = {}
                continue
            elif _opt == 'A':
                break
            else:
                raise Exception

            # IF USER DID NOT ABORT AND THERE ARE WORDS TO DELETE
            if len(TO_SUB_DICT) > 0:

                _str_replace = set([(k, v) for k,v in TO_SUB_DICT.items()])

                Trfm = TextReplacer(str_replace=_str_replace)
                self.CLEANED_TEXT =Trfm.fit_transform(self.CLEANED_TEXT)
                del Trfm

        del TO_SUB_DICT

        if converted:
            self.as_list_of_strs()
        del converted


    def as_list_of_lists(self):
        """
        Convert CLEANED_TEXT object to a possibly ragged vector of
        vectors, each vector containing split text.

        """


        if self.is_list_of_lists:
            pass
        elif not self.is_list_of_lists:  # MUST BE LIST OF strs
            # ASSUME THE TEXT STRING CAN BE SEPARATED ON ' '
            self.CLEANED_TEXT = \
                np.fromiter(map(str.split, self.CLEANED_TEXT), dtype=object)
            for row_idx in range(len(self.CLEANED_TEXT)):
                # WONT LET ME DO np.fromiter(map(np.ndarray SO PLUG-N-CHUG
                self.CLEANED_TEXT[row_idx] = \
                    np.array(self.CLEANED_TEXT[row_idx], dtype='<U30')

            self.is_list_of_lists = True


    def as_list_of_strs(self):
        """Convert CLEANED_TEXT object to a single vector of strings."""
        if not self.is_list_of_lists:
            pass
        elif self.is_list_of_lists:
            self.CLEANED_TEXT = \
                np.fromiter(map(' '.join, self.CLEANED_TEXT), dtype=object)
            self.is_list_of_lists = False


    def dump_to_csv(self):
        """Dump CLEANED_TEXT object to csv."""

        print(f'\nSaving CLEANED TEXT to csv...')

        super().dump_to_csv(self.CLEANED_TEXT)


    def dump_to_txt(self):
        """Dump CLEANED_TEXT object to txt."""

        print(f'\nSaving CLEANED TEXT to txt file...')

        super().dump_to_txt(self.CLEANED_TEXT)


    def toggle_undo(self):
        """Turn undo capability ON or OFF to save memory (only used in menu())."""
        if self.undo_status is False:   # CONVERTING OVER TO True
            self.undo_status = True
        elif self.undo_status is True:   # CONVERTING OVER TO False
            self.CLEANED_TEXT_BACKUP = None
            self.undo_status = False


    def lex_lookup(self, print_notes:Optional[bool] = False):

        """
        Scan entire CLEANED_TEXT object and prompt for handling of words
        not in LEXICON.


        Parameters
        ----------
        print_notes:
            Optional[bool], default=False - pizza


        """

        # PUT self.CLEANED_TEXT INTO MAJUSCULE AND list_of_list FORMAT IF NOT ALREADY

        converted = False
        if not self.is_list_of_lists:
            self.as_list_of_lists()
            converted = True


        print_notes = print_notes or False

        if print_notes: print(f'\nRunning _lexicon cross-reference...')

        Trfm = TextLookupRealTime(
            update_lexicon=self.update_lexicon,
            skip_numbers=True,
            auto_split=True,
            auto_add_to_lexicon=self.auto_add,
            auto_delete=self.auto_delete,
            remove_empty_rows=False,   # pizza gonna need to reckon with this
            verbose=print_notes
        )

        self.CLEANED_TEXT = Trfm.transform(self.CLEANED_TEXT)

        if converted:
            self.as_list_of_strs()
        del converted




































