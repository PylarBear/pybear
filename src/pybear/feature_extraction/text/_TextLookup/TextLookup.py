# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Optional, Sequence
from typing_extensions import Union
from ._type_aliases import XContainer, WipXContainer

from copy import deepcopy
import numbers

import pandas as pd
import polars as pl

from ._validation._validation import _validation

from ._transform._auto_word_splitter import _auto_word_splitter
from ._transform._manual_word_splitter import _manual_word_splitter
from ._transform._quasi_auto_word_splitter import _quasi_auto_word_splitter
from ._transform._view_snippet import _view_snippet
from ._transform._word_editor import _word_editor

from .._Lexicon.Lexicon import Lexicon

from ....data_validation import validate_user_input as vui

from ....base._copy_X import copy_X

from ....utilities._DictMenuPrint import DictMenuPrint





class TextLookup:

    """
    Pizza
    TO NEVER ALLOW IT TO GO INTO MANUAL MODE, SET EITHER auto_add_to_lexicon OR auto_delete
    (BUT NOT BOTH) to True.
    TO ALLOW ENTRY TO MANUAL MODE, BOTH auto_add_to_lexicon AND auto_delete MUST BE False.


    Parameters
    ----------
    update_lexicon:
        Optional[bool], default=False -
    auto_add_to_lexicon:
        Optional[bool], default=False - AUTOMATICALLY ADDS AN UNKNOWN
        WORD TO LEXICON_UPDATE # W/O PROMPTING USER (JUST GOES ALL THE
        WAY THRU WITHOUT PROMPTS) AUTOMATICALLY SENSES AND MAKES 2-WAY
        SPLITS
    auto_delete:
        Optional[bool], default=False -
    skip_numbers:
        Optional[bool], default=True -
    DELETE_ALWAYS:
        Optional[Union[Sequence[str], None]], default=None -
    REPLACE_ALWAYS:
        Optional[Union[dict[str, str], None]], default=None -
    SKIP_ALWAYS:
        Optional[Union[Sequence[str], None]], default=None -
    SPLIT_ALWAYS:
        Optional[Union[dict[str, Sequence[str]], None]], default=None -
    verbose:
        Optional[bool], default=False - display helpful information


    """


    def __init__(
        self,
        update_lexicon: Optional[bool] = False,
        skip_numbers: Optional[bool] = True,
        auto_split: Optional[bool] = True,
        auto_add_to_lexicon: Optional[bool] = False,
        auto_delete: Optional[bool] = False,
        DELETE_ALWAYS: Optional[Union[Sequence[str], None]] = None,
        REPLACE_ALWAYS: Optional[Union[dict[str, str], None]] = None,
        SKIP_ALWAYS: Optional[Union[Sequence[str], None]] = None,
        SPLIT_ALWAYS: Optional[Union[dict[str, Sequence[str]], None]] = None,
        verbose: Optional[bool] = False
    ) -> None:


        self.update_lexicon: bool = update_lexicon
        self.skip_numbers: bool = skip_numbers
        self.auto_split: bool = auto_split
        self.auto_add_to_lexicon: bool = auto_add_to_lexicon
        self.auto_delete: bool = auto_delete

        self.SKIP_ALWAYS: Sequence[str] = SKIP_ALWAYS
        self.SPLIT_ALWAYS: dict[str, Sequence[str]] = SPLIT_ALWAYS
        self.DELETE_ALWAYS: Sequence[str] = DELETE_ALWAYS
        self.REPLACE_ALWAYS: dict[str, str] = REPLACE_ALWAYS

        self.verbose = verbose

        self.LEXICON_ADDENDUM: list[str] = []
        self.KNOWN_WORDS: list[str] = deepcopy(Lexicon().lexicon_)


        _LEX_LOOK_DICT = {
            'a': 'Add to Lexicon',
            'e': 'Replace',
            'f': 'Replace always',
            'd': 'Delete',
            'l': 'Delete always',
            's': 'Split',
            'u': 'Split always',
            'k': 'Skip',
            'w': 'Skip always',
            'q': 'Quit'
        }

        if not self.update_lexicon:
            del _LEX_LOOK_DICT['A']

        self.LexLookupMenu = DictMenuPrint(
            _LEX_LOOK_DICT,
            disp_width=75,
            fixed_col_width=25
        )


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
            elif __ == 'N': continue
            elif __ == 'A': break

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



    def _display_lexicon_update(self, n=None):
        """Prints and returns LEXICON_ADDENDUM object for copy and paste into LEXICON."""
        if len(self.LEXICON_ADDENDUM) != 0:
            self.LEXICON_ADDENDUM.sort()
            print(f'LEXICON ADDENDUM:')
            print(f'[')
            for _ in self.LEXICON_ADDENDUM[:(n or len(self.LEXICON_ADDENDUM))]:
                print(f'    "{_}"{"" if _ == self.LEXICON_ADDENDUM[-1] else ","}')
            print(f']')
            print()
        else:
            print(f'*** EMPTY ***')


    def _split_or_replace_handler(
        self,
        _line: list[str],
        _word_idx: numbers.Integral,
        _NEW_WORDS: list[str]
    ) -> list[str]:

        _word = _line[_word_idx]

        _line.pop(_word_idx)

        # GO THRU _NEW_WORDS BACKWARDS
        for slot_idx, _new_word in range(len(_NEW_WORDS) - 1, -1, -1):
            _line.insert(_word_idx, _NEW_WORDS[slot_idx])

            if self.update_lexicon:
                # pizza think on SKIP_ALWAYS
                if _new_word in self.KNOWN_WORDS or _new_word in self.SKIP_ALWAYS:
                    continue

                if self.auto_add_to_lexicon:
                    self.LEXICON_ADDENDUM.append(_NEW_WORDS[slot_idx])
                    self.KNOWN_WORDS.append(_NEW_WORDS[slot_idx])
                    continue

                # if _new_words is not KNOWN or not skipped...
                print(f"\n*{_NEW_WORDS[slot_idx]}* IS NOT IN LEXICON\n")
                _ = self.LexLookupMenu.choose('Select option', allowed='akw')
                if _ == 'a':
                    self.LEXICON_ADDENDUM.append(_NEW_WORDS[slot_idx])
                    self.KNOWN_WORDS.append(_NEW_WORDS[slot_idx])
                elif _ == 'k':
                    pass
                elif _ == 'w':
                    self.SKIP_ALWAYS.append(_word)
                else:
                    raise Exception

        del _NEW_WORDS

        return _line


    def lex_lookup(
        self,
        X: XContainer,
        copy: Optional[bool] = True
    ):

        """
        Scan tokens in X and prompt for handling of tokens not in Lexicon.


        Parameters
        ----------
        X:
            XContainer - The data in (possibly ragged) 2D array-like format.
        copy:
            Optional[bool], default=True - whether to operate directly on X.


        Return
        ------
        -
            XContainer

        """

        # VALIDATION ###################################################

        _validation(
            X,
            self.update_lexicon,
            self.skip_numbers,
            self.auto_split,
            self.auto_add_to_lexicon,
            self.auto_delete,
            self.DELETE_ALWAYS,
            self.REPLACE_ALWAYS,
            self.SKIP_ALWAYS,
            self.SPLIT_ALWAYS,
            self.verbose
        )

        self.DELETE_ALWAYS = self.DELETE_ALWAYS or []
        self.REPLACE_ALWAYS = self.REPLACE_ALWAYS or {}
        self.SKIP_ALWAYS = self.SKIP_ALWAYS or []
        self.SPLIT_ALWAYS = self.SPLIT_ALWAYS or {}

        # END VALIDATION ###############################################

        if copy:
            _X = copy_X(X)
        else:
            _X = X

        # convert X to list-of-lists -- -- -- -- -- -- -- -- -- -- -- --
        # we know from validation it is legit 2D
        if isinstance(_X, pd.DataFrame):
            _X = list(map(list, _X.values))
        elif isinstance(_X, pl.DataFrame):
            _X = list(map(list, _X.rows()))
        else:
            _X = list(map(list, _X))
        # END convert X to list-of-lists -- -- -- -- -- -- -- -- -- -- --

        _X: WipXContainer


        # MANAGE THE CONTENTS OF LEXICON ADDENDUM -- -- -- -- -- -- -- --
        _abort = False

        if self.update_lexicon and len(self.LEXICON_ADDENDUM) != 0:

            print(f'\n*** LEXICON ADDENDUM IS NOT EMPTY ***\n')
            print(f'LEXICON ADDENDUM has {len(self.LEXICON_ADDENDUM)} entries')
            print(f'First 10 in LEXICON ADDENDUM:')
            self._display_lexicon_update(n=10)
            print()

            _ = vui.validate_user_str(
                f'Empty it(e), Proceed anyway(p), Abort TextLookup(a) > ',
                'AEP'
            )
            if _ == 'A':
                _abort = True
            elif _ == 'E':
                self.LEXICON_ADDENDUM = []
            elif _ == 'P':
                pass
            del _
        # END MANAGE THE CONTENTS OF LEXICON ADDENDUM -- -- -- -- -- -- --

        if self.verbose:
            print(f'\nRunning Lexicon cross-reference...')

        _quit = False
        n_edits = 0
        word_counter = 0
        total_words = sum(map(len, _X))
        for row_idx in [range(len(_X)) if not _abort else []][0]:

            if self.verbose:
                print(f'\nStarting row {row_idx+1} of {len(_X)}')
                print(f'\nCurrent state of ')
                self._display_lexicon_update()

            # GO THRU BACKWARDS BECAUSE A SPLIT OR DELETE WILL CHANGE THE ARRAY OF WORDS
            for word_idx in range(len(_X[row_idx]) - 1, -1, -1):

                # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
                # Manage in-situ save option if in manual edit and updating lexicon
                if self.update_lexicon and not self.auto_add_to_lexicon and not self.auto_delete and n_edits % 10 == 0:

                    _prompt = f'Save all that hard work to file(s) or continue(c) > '
                    if vui.validate_user_str(_prompt, 'SC') == 'S':
                        _prompt2 = f'Save to csv(c) or txt(t)? > '
                        if vui.validate_user_str(_prompt2, 'CT') == 'C':
                            self.dump_to_csv()
                        else:
                            self.dump_to_txt()
                        del _prompt2

                    del _prompt
                # END manage in-situ save -- -- -- -- -- -- -- -- -- -- -- --

                word_counter += 1
                if self.verbose and word_counter % 1000 == 0:
                    print(f'Running word {word_counter:,} of {total_words:,}...')

                word = _X[row_idx][word_idx]

                # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
                # short-circuit for things already known or learned in-situ
                if word in self.KNOWN_WORDS:
                    if self.verbose:
                        print(f'\n*** {word} IS ALREADY IN LEXICON ***\n')
                    continue

                if word in self.DELETE_ALWAYS:
                    if self.verbose:
                        print(f'\n*** DELETING {word} ***\n')
                    _X[row_idx].pop(word_idx)
                    continue

                if word in self.REPLACE_ALWAYS:
                    if self.verbose:
                        print(f'\n*** ALWAYS REPLACE {word} WITH {self.REPLACE_ALWAYS[word]} ***\n')
                    _X[row_idx][word_idx] = self.REPLACE_ALWAYS[word]
                    continue

                if word in self.SKIP_ALWAYS:
                    if self.verbose:
                        print(f'\n*** ALWAYS SKIP {word} ***\n')
                    continue

                if word in self.SPLIT_ALWAYS:
                    # this may have had words in it from the user at instantiation
                    _X[row_idx].pop(word_idx)
                    for slot_idx in range(len(self.SPLIT_ALWAYS[word]) - 1, -1, -1):
                        # GO THRU NEW_WORDS BACKWARDS
                        _X[row_idx].insert(word_idx, self.SPLIT_ALWAYS[word][slot_idx])
                    continue

                # short circuit for numbers
                if self.skip_numbers:
                    try:
                        float(word)
                        # if get to here its a number, go to next word
                        continue
                    except:
                        pass
                # END short circuit for numbers

                # END short-circuit for things already known or learned in-situ
                # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

                # short-circuit for auto-split -- -- -- -- -- -- -- -- --
                # last ditch before auto-add & auto-delete, try to save the word
                # LOOK IF word IS 2 KNOWN WORDS MOOSHED TOGETHER
                # LOOK FOR FIRST VALID SPLIT IF len(word) >= 4
                if self.auto_split and len(word) >= 4:
                    _NEW_LINE = _auto_word_splitter(
                        word, word_idx, _X[word_idx], self.KNOWN_WORDS, self.verbose
                    )
                    if any(_NEW_LINE):
                        _X[row_idx] = _NEW_LINE
                        # since auto_word_splitter requires that both halves already
                        # be in the Lexicon, just continue to the next word
                        del _NEW_LINE
                        continue
                    # else: if _NEW_LINE is empty, there wasnt a valid split, just
                    # pass, if auto_delete is True, that will delete this word and
                    # go to the next word. if auto_delete is False, it will also
                    # pass thru quasi_auto_splitter, then the user will have to
                    # deal with it manually.
                    del _NEW_LINE
                # END short-circuit for auto-split -- -- -- -- -- -- -- --

                # short-circuit for auto-add -- -- -- -- -- -- -- -- -- --
                # ANYTHING that is in X that is not in Lexicon gets added
                # and stays in X.
                if self.update_lexicon and self.auto_add_to_lexicon:
                    if self.verbose:
                        print(f'\n*** AUTO-ADDING {word} TO LEXICON ADDENDUM ***\n')
                    self.LEXICON_ADDENDUM.append(word)
                    self.KNOWN_WORDS.append(word)

                    continue
                # END short-circuit for auto-add -- -- -- -- -- -- -- --

                # short-circuit for auto-delete -- -- -- -- -- -- -- -- -- --
                if self.auto_delete:
                    if self.verbose:
                        print(f'\n*** AUTO-DELETING {word} ***\n')
                    _X[row_idx].pop(word_idx)
                    continue
                # END short-circuit for auto-delete -- -- -- -- -- -- --

                # after here it is implicit not auto_add_to_lexicon and not auto_delete

                # v v v MANUAL MODE v v v v v v v v v v v v v v v v v v v v v
                # implicit not auto_and and not auto_delete
                # word is not in KNOWN_WORDS or any repetitive operation holders

                # an edit is guaranteed to happen after this point
                n_edits += 1

                # quasi-automate split recommendation -- -- -- -- -- -- -- -- --
                # if we had auto_split=True and we get to here, its because there
                # were no valid splits and just passed thru, so the word will also
                # pass thru here. if auto_split was False and we get to here,
                # we are about to enter manual mode. the user is forced into this
                # as a convenience to partially automate the process of finding
                # splits as opposed to having to manually type 2-way splits
                # over and over.

                _continue = False
                if len(word) >= 4:
                    _NEW_LINE = _quasi_auto_word_splitter(
                        word, word_idx, _X[row_idx], self.KNOWN_WORDS, self.verbose
                    )
                    # if the user did not opt to take any of splits (or if there
                    # werent any), then _NEW_LINE is empty, and the user is
                    # forced into the big manual menu.
                    if any(_NEW_LINE):
                        _X[row_idx] = _NEW_LINE
                        # since quasi_auto_word_splitter requires that both halves
                        # already be in the Lexicon, just continue to the next word
                        del _NEW_LINE
                        continue

                    del _NEW_LINE
                # END quasi-automate split recommendation -- -- -- -- -- -- --

                print(_view_snippet(_X[row_idx], word_idx, _span=9))
                print(f"\n*{word}* IS NOT IN LEXICON\n")
                _selection = self.LexLookupMenu.choose('Select option')

                if _selection == 'a':    # 'a': 'Add to Lexicon'
                    self.LEXICON_ADDENDUM.append(word)
                    self.KNOWN_WORDS.append(word)
                    # and X is unchanged
                elif _selection == 'd':   # 'd': 'Delete'
                    _X[row_idx].pop(word_idx)
                elif _selection in 'ef':   # 'e': 'Replace', 'f': 'Replace always',
                    new_word = _word_editor(word, _prompt=f'Enter new word to replace *{word}*')

                    if _selection == 'f':
                        self.REPLACE_ALWAYS[word] = new_word

                    _X[row_idx] = self._split_or_replace_handler(_X[row_idx], word_idx, [new_word])
                elif _selection == 'l':   # 'l': 'Delete always'
                    # DELETE CURRENT ENTRY IN CURRENT ROW
                    _X[row_idx].pop(word_idx)
                    # PUT WORD INTO DELETE_ALWAYS_LIST
                    self.DELETE_ALWAYS.append(word)
                elif _selection == 'k':   # 'k': 'Skip'
                    pass
                elif _selection == 'w':   # 'w': 'Skip always'
                    self.SKIP_ALWAYS.append(word)
                elif _selection in 'su':   # 's': 'Split', 'u': 'Split always'
                    # this split is different than auto and quasi... those split
                    # on both halves of the original word being in Lexicon, but
                    # here the user might pass something new, so this needs to
                    # run thru _split_or_replace_handler in case update_lexicon
                    # is True and the new words arent in the Lexicon
                    _NEW_WORDS = _manual_word_splitter(word, word_idx, _X[row_idx], self.KNOWN_WORDS, self.verbose)
                    if _selection == 'u':
                        self.SPLIT_ALWAYS[word] = _NEW_WORDS
                    if any(_NEW_WORDS):
                        _X[row_idx] = self._split_or_replace_handler(_X[row_idx], word_idx, _NEW_WORDS)
                    del _NEW_WORDS
                elif _selection == 'q':   # 'q': 'Quit'
                    _quit = True
                    break
                else:
                    raise Exception

            if _quit:
                break

        del n_edits, word_counter

        if self.verbose:
            print(f'\n*** LEX LOOKUP COMPLETE ***\n')

        if self.update_lexicon and not _abort:
            # show this to the user so they can copy-paste into Lexicon
            if len(self.LEXICON_ADDENDUM) != 0:
                print(f'\n*** COPY AND PASTE THESE WORDS INTO LEXICON ***\n')
                self._display_lexicon_update()

        del _abort
















