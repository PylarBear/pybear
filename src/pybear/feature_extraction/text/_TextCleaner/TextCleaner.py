# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import sys, inspect, math, warnings
from typing import Iterable
import numpy as np, pandas as pd
# PIZZA NEED PLOTLY OR MATPLOTLIB
from ....utilities._get_module_name import get_module_name
from .._Lexicon.Lexicon import Lexicon as lex
from ....data_validation import (
    validate_user_input as vui,
    arg_kwarg_validater
)
from .. import (
    alphanumeric_str as ans,
    _stop_words as sw,
    _statistics as stats
)


# _exception                Exception handling for this module.
# menu
# delete_empty_rows         Remove textless rows from data.
# remove_characters         Keep only allowed or removed disallowed characters from entire CLEANED_TEXT object.
# _strip                    Remove multiple spaces and leading and trailing spaces from all text in CLEAND_TEXT object.
# normalize                 Set all text in CLEANED_TEXT object to upper case (default) or lower case.
# return_row_uniques        Return a potentially ragged vector containing the unique words for each row in CLEANED_TEXT object.
# view_cleaned_text         Print cleaned text to screen.
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
# lex_lookup                Scan entire CLEANED_TEXT object and prompt for handling of words not in LEXICON.  <<<<<
# END STUFF FOR LEXICON LOOKUP #####################################################################################



class TextCleaner:

    def __init__(
        self,
        LIST_OF_STRINGS: Iterable[str],
        update_lexicon: bool = False,
        auto_add: bool = False,
        auto_delete: bool = False
    ) -> None:  # return_as_list_of_lists=False,  # pizza what this mean?

        """
        pizza say something

        Parameters
        ----------
        LIST_OF_STRINGS:
            Iterable[str] -
        update_lexicon:
            bool=False -
        auto_add:
            bool=False -
        auto_delete:
            bool=False

        # pizza
        # LIST_OF_STRINGS MUST BE PASSED AS ['str1','str2','str3'...],
        # JUST LIKE NNLM50

        # auto_add AUTOMATICALLY ADDS AN UNKNOWN WORD TO LEXICON_UPDATE
        # W/O PROMPTING USER
        # (JUST GOES ALL THE WAY THRU WITHOUT PROMPTS) AUTOMATICALLY
        # SENSES AND MAKES 2-WAY SPLITS



        Return
        ------
        -
            None.

        """




        self.this_module = get_module_name(str(sys.modules[__name__]))

        fxn = "__init__"

        ################################################################
        # ARG / KWARG VALIDATION #######################################

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            LIST_OF_STRINGS = np.array(LIST_OF_STRINGS)
            if len(LIST_OF_STRINGS.shape) == 1:
                LIST_OF_STRINGS = LIST_OF_STRINGS.reshape((1, -1))

        try:
            self.LIST_OF_STRINGS = \
                np.fromiter(map(str, LIST_OF_STRINGS[0]), dtype='<U10000')
        except:
            raise TypeError(f"LIST_OF_STRINGS MUST CONTAIN DATA THAT CAN "
                            f"BE CONVERTED TO str", fxn)

        self.CLEANED_TEXT = self.LIST_OF_STRINGS.copy().astype(object)
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

        # 24_06_23 pizza these probably need to stay as attrs
        self.LEXICON_ADDENDUM = [np.empty((1,0), dtype='<U30')[0]]
        self.KNOWN_WORDS = None

        self.LEX_LOOK_DICT = {
            'A': 'ADD TO LEXICON', 'D': 'DELETE', 'E': 'EDIT', 'F': 'EDIT ALL',
            'L': 'DELETE ALL', 'S': 'SPLIT', 'U': 'SPLIT ALWAYS', 'K': 'SKIP ONCE',
            'W': 'SKIP ALWAYS', 'Y': 'VIEW LEXICON ADDENDUM', 'Z': 'GO TO MAIN MENU'
        }  # DONT USE 'T' !!!

        if not self.update_lexicon:
            del self.LEX_LOOK_DICT['A']

        self.LEX_LOOK_DISPLAY = ", ".join(list(f'{v}({k.lower()})' for k, v in self.LEX_LOOK_DICT.items()))
        self.lex_look_allowed = "".join(list(self.LEX_LOOK_DICT.keys())).lower()
        self.undo_status = False
        self.CLEANED_TEXT_BACKUP = None

        self.MENU_DICT = {
            'D': {'label':'delete_empty_rows',                              'function': self.delete_empty_rows        },
            'R': {'label':'remove_characters',                              'function': self.remove_characters        },
            'S': {'label':'strip',                                          'function': self._strip                   },
            'N': {'label':'normalize',                                      'function': self.normalize                },
            'A': {'label':'statistics',                                     'function': self.statistics               },
            'K': {'label':'word_counter',                                   'function': self.word_counter             },
            'V': {'label':'view_CLEANED_TEXT',                              'function': self.view_cleaned_text        },
            'U': {'label':'view_row_uniques',                               'function': self.view_row_uniques         },
            'O': {'label':'view_overall_uniques',                           'function': self.view_overall_uniques     },
            'Y': {'label':'view_lexicon_addendum',                          'function': self.display_lexicon_update   },
            'T': {'label':'remove_stops',                                   'function': self.remove_stops             },
            'J': {'label':'justify',                                        'function': self.justify                  },
            'W': {'label':'delete_words',                                   'function': self.delete_words             },
            'B': {'label':'substitute_words',                               'function': self.substitute_words         },
            'L': {'label':'as_list_of_lists',                               'function': self.as_list_of_lists         },
            'I': {'label':'as_list_of_strs',                                'function': self.as_list_of_strs          },
            'P': {'label':'lex_lookup',                                     'function': self.lex_lookup               },
            'C': {'label':'dump_to_csv',                                    'function': self.dump_to_csv              },
            'X': {'label':'dump_to_txt',                                    'function': self.dump_to_txt              },
            'E': {'label':f'toggle UNDO (currently {self.undo_status})',    'function': self.toggle_undo              },
            'F': {'label':'undo',                                           'function': None                          },
            'Q': {'label':'quit',                                           'function': None                          },
            'Z': {'label':'accept and exit',                                'function': None                          }
        }

        # END DECLARATIONS #############################################
        ################################################################

    def _exception(self, words, fxn=None):
        """Exception handling for this module."""
        fxn = f".{fxn}()" if not fxn is None else ""
        raise Exception(f'\n*** {self.this_module}{fxn} >>> {words} ***\n')





    def menu(self, allowed:str=None, disallowed:str=None):




        # VALIDATION ###################################################
        fxn = inspect.stack()[0][3]

        allowed_key = "".join(self.MENU_DICT.keys()).upper()
        # alloweds = 'ABCDEFIJLNOPQRSTUWXZ'

        if not allowed is None and not disallowed is None:
            self._exception(f'{fxn} >>> CANNOT ENTER BOTH allowed AND disallowed, MUST BE ONE OR THE OTHER OR NEITHER', fxn=fxn)
        elif not allowed is None:
            if not isinstance(allowed, str):
                self._exception(f'{fxn} allowed KWARG REQUIRES str AS INPUT', fxn)
            for _char in allowed:
                if not _char in allowed_key:
                    self._exception(f'INVALID KEY "{_char}" IN allowed, MUST BE IN {allowed_key}.', fxn)
        elif not disallowed is None:
            if not isinstance(disallowed, str):
                self._exception(f'{fxn} disallowed KWARG REQUIRES str AS INPUT', fxn)
            for _char in disallowed:
                if not _char in allowed_key:
                    self._exception(f'INVALID KEY "{_char}" IN disallowed, MUST BE IN {allowed_key}.', fxn)
            allowed = ''.join([_ for _ in allowed_key if not _ in disallowed])
        elif allowed is None and disallowed is None:
            allowed = allowed_key

        allowed = allowed.upper()

        for _ in allowed:
            if _ not in allowed_key:
                self._exception(f'{fxn} allowed KWARG CHARACTERS MUST BE IN *{allowed_key}*', fxn)
        # END VALIDATION ###############################################

        while True:

            # BUILD MENU DISPLAY #######################################
            # MUST BE INSIDE while BECAUSE PRINTOUT IS DYNAMIC BASED ON undo_status AND AVAILABILITY OF BACKUP
            # MANAGE AVAILABILITY OF undo COMMAND BASED ON STATUS OF undo_status AND BACKUP
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
                self._exception(f'USER TERMINATED', fxn)
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
            allowed_chars_as_str:str=ans.alphanumeric_str(),
            disallowed_chars_as_str:str=None
        ):






        # 24_06_22 see the benchmark module for this.
        # winner was map_set

        """Keep only allowed or removed disallowed characters from entire CLEANED_TEXT object."""
        fxn = inspect.stack()[0][3]

        # VALIDATION ###################################################
        if not allowed_chars_as_str is None:
            if not 'STR' in str(type(allowed_chars_as_str)).upper():
                self._exception(f'allowed_chars_as_str MUST BE GIVEN AS str', fxn)
            if not disallowed_chars_as_str is None:
                self._exception(f'CANNOT ENTER BOTH allowed_chars AND disallowed_chars. ONLY ONE OR THE OTHER OR NEITHER.', fxn)
        elif allowed_chars_as_str is None and disallowed_chars_as_str is None:
            self._exception(f'MUST SPECIFY ONE OF allowed_chars AND disallowed_chars.', fxn)
        elif allowed_chars_as_str is None:
            if not 'STR' in str(type(disallowed_chars_as_str)).upper():
                self._exception(f'disallowed_chars_as_str MUST BE GIVEN AS str', fxn)
        # END VALIDATION ###############################################

        if not self.is_list_of_lists:   # MUST BE LIST OF strs
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
                UNIQUES = "".join(np.unique(np.fromiter((_ for _ in " ".join(self.CLEANED_TEXT[row_idx])), dtype='<U1')))
                for char in UNIQUES:
                    if (not allowed_chars_as_str is None and char not in allowed_chars_as_str) or \
                            (not disallowed_chars_as_str is None and char in disallowed_chars_as_str):
                        self.CLEANED_TEXT[row_idx] = np.char.replace(self.CLEANED_TEXT[row_idx], char, '')

                if not f'' in self.CLEANED_TEXT[row_idx]:
                    continue
                else:
                    self.CLEANED_TEXT[row_idx] = self.CLEANED_TEXT[row_idx][..., self.CLEANED_TEXT[row_idx]!='']

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
                        map(str.strip, self.CLEANED_TEXT[row_idx]); break
            elif not self.is_list_of_lists:   # MUST BE LIST OF strs
                while True:
                    if f'  ' in self.CLEANED_TEXT[row_idx]:
                        self.CLEANED_TEXT[row_idx] = str(np.char.replace(self.CLEANED_TEXT[row_idx], f'  ', f' '))
                    else:
                        self.CLEANED_TEXT[row_idx] = self.CLEANED_TEXT[row_idx].strip(); break


    def normalize(self, upper=True):    # IF NOT upper THEN lower
        """Set all text in CLEANED_TEXT object to upper case (default) or lower case."""




        # WILL PROBABLY BE A RAGGED ARRAY AND np.char WILL THROW A FIT, SO GO ROW BY ROW
        if self.is_list_of_lists:
            for row_idx in range(len(self.CLEANED_TEXT)):
                if upper:
                    self.CLEANED_TEXT[row_idx] = np.fromiter(map(str.upper, self.CLEANED_TEXT[row_idx]), dtype='U30')
                elif not upper:
                    self.CLEANED_TEXT[row_idx] = np.fromiter(map(str.lower, self.CLEANED_TEXT[row_idx]), dtype='U30')
        elif not self.is_list_of_lists:   # LIST OF strs
            if upper:
                self.CLEANED_TEXT = np.fromiter(map(str.upper, self.CLEANED_TEXT), dtype='U100000')
            elif not upper:
                self.CLEANED_TEXT = np.fromiter(map(str.lower, self.CLEANED_TEXT), dtype='U100000')


    def view_cleaned_text(self):
        """Print cleaned text to screen."""

        _type = "LISTS" if self.is_list_of_lists else "STRINGS"

        print(f'\nCLEANED TEXT (currently in memory as {_type}):')

        del _type

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


        elif return_counts:
            UNIQUES_HOLDER = np.empty(len(self.CLEANED_TEXT), dtype=object)
            COUNTS_HOLDER = np.empty(len(self.CLEANED_TEXT), dtype=object)
            for row_idx in range(len(self.CLEANED_TEXT)):
                UNIQUES_HOLDER[row_idx], COUNTS_HOLDER[row_idx] = \
                    np.unique(self.CLEANED_TEXT[row_idx], return_counts=True)


        # CHANGE BACK TO LIST OF strs
        if converted:
            self.as_list_of_strs()
        del converted


        if not return_counts:

            return UNIQUES

        elif return_counts:

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
            return_counts = {'Y':True, 'N':False}[vui.validate_user_str(f'View counts? (y/n) > ', 'YN')]

        if return_counts is True:
            UNIQUES, COUNTS = self.return_row_uniques(return_counts=True)
        elif return_counts is False:
            UNIQUES = self.return_row_uniques(return_counts=False)

        for row_idx in range(len(UNIQUES)):
            print(f'ROW {row_idx+1}:')
            if return_counts is True:
                [print(f'   {UNIQUES[row_idx][word_idx]}'.ljust(30) + f'{COUNTS[row_idx][word_idx]}') for word_idx in range(len(UNIQUES[row_idx]))]
            elif return_counts is False:
                [print(f'   {UNIQUES[row_idx][word_idx]}') for word_idx in range(len(UNIQUES[row_idx]))]
            print()

        if return_counts is True:
            del UNIQUES, COUNTS
        elif return_counts is False:
            del UNIQUES


    def return_overall_uniques(self, return_counts=False):
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

            if converted:
                self.as_list_of_strs()
            del converted

            return UNIQUES, COUNTS


    def view_overall_uniques(self, return_counts:bool=None):
        """
        Print overall uniques and optionally counts to screen.


        """

        fxn = inspect.stack()[0][3]

        return_counts = arg_kwarg_validater(
            return_counts,
            'return_counts',
            [True, False, None],
            self.this_module,
            fxn
        )

        if return_counts is None:
            return_counts = {'Y':True, 'N':False}[vui.validate_user_str(f'View counts? (y/n) > ', 'YN')]

        if return_counts is True:
            UNIQUES, COUNTS = self.return_overall_uniques(return_counts=True)

            MASK = np.flip(np.argsort(COUNTS))
            UNIQUES = UNIQUES[..., MASK]
            COUNTS = COUNTS[..., MASK]
            del MASK

            print(f'OVERALL UNIQUES:')
            for idx in range(len(UNIQUES)):
                print(f'   {UNIQUES[idx]}'.ljust(30) + f'{COUNTS[idx]}')
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
            for word_idx in range(len(self.CLEANED_TEXT[row_idx])-1, -1, -1):
                if self.CLEANED_TEXT[row_idx][word_idx] in sw._stop_words():
                    self.CLEANED_TEXT[row_idx] = np.delete(self.CLEANED_TEXT[row_idx], word_idx, axis=0)
                if len(self.CLEANED_TEXT) == 0:
                    break

        if converted:
            self.as_list_of_strs()
        del converted


    def justify(self, chars:int=None):
        """Fit text as strings or as lists to user-specified number of characters per row."""

        # ALSO SEE text.notepad_justifier FOR SIMILAR CODE, IF EVER CONSOLIDATING

        fxn = inspect.stack()[0][3]

        if not chars is None:
            chars = arg_kwarg_validater(
                chars,
                'characters',
                list(range(30,50001)),
                self.this_module,
                fxn
            )
        elif chars is None:
            # DONT PUT THIS IN akv(return_if_none=)... PROMPTS USER FOR
            # sINPUT BEFORE ENDING args/kwargs TO akv
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
                    seed += new_word+' '
                elif len(seed) + len(new_word) > max_line_len:
                    NEW_TXT = np.insert(NEW_TXT, len(NEW_TXT), seed.strip(), axis=0)
                    seed = new_word+' '
        if len(seed) > 0: NEW_TXT = np.insert(NEW_TXT, len(NEW_TXT), seed.strip(), axis=0)

        del max_line_len, seed, new_word

        self.CLEANED_TEXT = NEW_TXT
        del NEW_TXT
        self.is_list_of_lists = False


        # OBJECT WAS WORKED ON AS LIST OF LISTS, BUT OUTPUT IS LIST OF STRS
        if converted:
            pass  # MEANING THAT IS WAS list_of_strs TO START WITH, JUST LEAVE AS IS
        elif not converted:   # OTHERWISE WAS LIST OF LISTS TO START, SO CONVERT BACK TO LIST OF LISTS
            self.as_list_of_lists()
            map(str.strip, self.CLEANED_TEXT)
        del converted


    def delete_words(self):
        """Delete one or more words from the entire CLEANED_TEXT object."""

        converted = False
        if not self.is_list_of_lists:
            self.as_list_of_lists()
            converted = True

        TO_DELETE = np.empty((1,0), dtype='<U30')[0]

        ctr = 0
        while True:
            while True:
                to_delete = input(f'\nEnter word to delete '
                    f'({[f"type *d* when done, " if ctr>0 else ""][0]}*z* to abort) > ').upper()
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
                TO_DELETE = np.empty((1,0), dtype='<U30')[0]; ctr=0

        if to_delete != 'Z' and len(TO_DELETE) > 0:  # IF USER DID NOT ABORT AND THERE ARE WORDS TO DELETE, PROCEED WITH DELETE

            for row_idx in range(len(self.CLEANED_TEXT)):
                for word_idx in range(len(self.CLEANED_TEXT[row_idx])-1, -1, -1):
                    if self.CLEANED_TEXT[row_idx][word_idx] in TO_DELETE:
                        self.CLEANED_TEXT[row_idx] = np.delete(self.CLEANED_TEXT[row_idx], word_idx, axis=0)

                    if len(self.CLEANED_TEXT[row_idx]) == 0:
                        break

        del TO_DELETE, to_delete

        if converted:
            self.as_list_of_strs()
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
                                 f'({["type *d* when done, " if ctr>0 else ""][0]}*z* to abort) > ').upper()
                if replaced in 'DZ':
                    break
                else:
                    replacement = input(f'Enter word to substitute in '
                                f'({["type *d* when done, " if ctr>0 else ""][0]}*z* to abort) > ').upper()
                    if replacement in 'DZ':
                        break
                if vui.validate_user_str(
                        f'User entered to replace *{replaced}* with *{replacement}*--- Accept? (y/n) > ',
                        'YN') == 'Y':
                    ctr += 1
                    TO_SUB_DICT[replaced] = replacement
            if replaced == 'Z' or replacement == 'Z':
                del ctr
                break
            print(f'\nUser entered to replace')
            [print(f'{k} with {v}') for k,v in TO_SUB_DICT.items()]
            if vui.validate_user_str(f'\nAccept? (y/n) > ', 'YN') == 'Y':
                del ctr
                break

        # IF USER DID NOT ABORT AND THERE ARE WORDS TO DELETE, PROCEED WITH DELETE
        if (replaced != 'Z' and replacement != 'Z') and len(TO_SUB_DICT) > 0:

            for row_idx in range(len(self.CLEANED_TEXT)):
                for word_idx in range(len(self.CLEANED_TEXT[row_idx])-1, -1, -1):
                    word = self.CLEANED_TEXT[row_idx][word_idx]
                    if word in TO_SUB_DICT:
                        self.CLEANED_TEXT[row_idx][word_idx] = TO_SUB_DICT[word]
            del word

        del TO_SUB_DICT, replaced, replacement

        if converted:
            self.as_list_of_strs()
        del converted


    # pizza dont forget to change all the as_list_of_lists and as_list_of_strs










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
        """Calculate frequencies for CLEANED_TEXT and print to screen or dump to file.
        """






        is_asc = True
        SUB_MENU = {
            'e': 'exit', 'c': 'set count cutoff', 'd': 'dump to file',
            'p': 'print table to screen', 'r': 'print chart to screen',
            's': f'change sort (currently XXX)'
        }  # DONT FORGET 's' BELOW

        cutoff_ct = 100 # SEED
        allowed = "".join(list(SUB_MENU.keys()))
        MASTER_UNIQUES, MASTER_COUNTS = self.return_overall_uniques(return_counts=True)   # DEFAULTS TO ASCENDING
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

            if selection == 'e': #'exit'
                break
            elif selection == 'c': #'set count cutoff'
                cutoff_ct = vui.validate_user_int(f'\nSet count cutoff (currently {cutoff_ct}) > ', min=1)
                MASK = np.where(MASTER_COUNTS >= cutoff_ct, True, False)
                WIP_UNIQUES = MASTER_UNIQUES[MASK]
                WIP_COUNTS = MASTER_COUNTS[MASK]
                del MASK
            elif selection == 'd': #'dump to file'
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
            elif selection == 'p': #'print table to screen'
                _pad = lambda x: f' ' * 5 + str(x)
                __ = 10
                print(_pad(f'WORD').ljust(2 * __) + f'FREQUENCY')
                for i in range(len(WIP_UNIQUES)):
                    print(f'{_pad(WIP_UNIQUES[i])}'.ljust(2 * __) + f'{WIP_COUNTS[i]}')
                print()
            elif selection == 'r': #'print chart to screen'
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
            elif selection == 's': #'change sort (currently XXX)'
                if is_asc:
                    is_asc = False
                elif not is_asc:
                    is_asc = True
                WIP_UNIQUES = np.flip(WIP_UNIQUES)
                MASTER_UNIQUES = np.flip(MASTER_UNIQUES)
                WIP_COUNTS = np.flip(WIP_COUNTS)
                MASTER_COUNTS = np.flip(MASTER_COUNTS)

        del SUB_MENU, display, selection, cutoff_ct, is_asc, MASTER_UNIQUES, MASTER_COUNTS
        try: del WIP_UNIQUES
        except: pass
        try: del WIP_COUNTS
        except: pass


    def dump_to_file_wrapper(self, core_write_function, _ext, kwargs):

        """
        Wrapper function for dumping CLEANED_TEXT object to csv or txt.


        """





        converted = False
        if self.is_list_of_lists:
            self.as_list_of_strs()
            converted = True

        while True:
            file_name = input(f'Enter filename > ')
            __ = vui.validate_user_str(f'User entered *{file_name}*  ---  Accept? (y) (n) (a)bort > ', 'YNA')
            if __ == 'Y':
                core_write_function(file_name+_ext, **kwargs)
                print(f'\n*** Dump to {_ext} successful. ***\n')
                break
            elif __ == 'N':
                continue
            elif __ == 'A':
                break

        if converted:
            self.as_list_of_lists()
        del converted


    def dump_to_csv(self):
        """Dump CLEANED_TEXT object to csv."""

        print(f'\nSaving CLEANED TEXT to csv...')

        converted = False
        if self.is_list_of_lists:
            self.as_list_of_strs()
            converted = True
        _core_fxn = pd.DataFrame(data=self.CLEANED_TEXT.transpose(), columns=[f'CLEANED_DATA']).to_csv

        self.dump_to_file_wrapper(_core_fxn, f'.csv', {'header':True, 'index':False})

        if converted:
            self.as_list_of_lists()
        del converted


    def dump_to_txt(self):
        """Dump CLEANED_TEXT object to txt."""

        print(f'\nSaving CLEANED TEXT to txt file...')

        def _core_fxn(full_path):   # DONT PUT kwargs OR **kwargs IN ()!
            with open(full_path, 'w') as f:
                for line in self.CLEANED_TEXT:
                    f.write(line + '\n')
                f.close()

        self.dump_to_file_wrapper(_core_fxn, f'.txt', {})


    def toggle_undo(self):
        """Turn undo capability ON or OFF to save memory (only used in menu())."""
        if self.undo_status is False:   # CONVERTING OVER TO True
            self.undo_status = True
        elif self.undo_status is True:   # CONVERTING OVER TO False
            self.CLEANED_TEXT_BACKUP = None
            self.undo_status = False



    ####################################################################
    ####################################################################
    # STUFF FOR LEXICON LOOKUP #########################################


    def lex_lookup(self, print_notes:bool=False):
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

        pizza, pizza, pizza = \
            _lex_lookup(
                flour,
                yeast,
                water,
                sauce,
                cheese,
                garlic,
                pepperoni
            )



    # END STUFF FOR LEXICON LOOKUP #####################################
    ####################################################################
    ####################################################################





































