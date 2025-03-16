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


    def _display_lexicon_update(
        self,
        n=None
    ) -> None:

        """
        Prints LEXICON_ADDENDUM object for copy and paste into Lexicon.


        Parameters
        ----------
        n:
            Optional[Union[int, None]], default=None - the number of
            entries in LEXICON_ADDENDUM to print.


        Return
        ------
        -
            None

        """

        print(f'LEXICON ADDENDUM:')
        if len(self.LEXICON_ADDENDUM) == 0:
            print(f'*** EMPTY ***')
        else:
            self.LEXICON_ADDENDUM.sort()
            print(f'[')
            for _ in self.LEXICON_ADDENDUM[:(n or len(self.LEXICON_ADDENDUM))]:
                print(f'    "{_}"{"" if _ == self.LEXICON_ADDENDUM[-1] else ","}')
            print(f']')
            print()


    def _split_or_replace_handler(
        self,
        _line: list[str],
        _word_idx: numbers.Integral,
        _NEW_WORDS: list[str]
    ) -> list[str]:

        """
        Handle removing an old word from a line, substituting in the new
        word(s), updating

        This is called after split, split always, replace, replace always.


        Parameters
        ----------
        _line:
            list[str] - the full line of the data that holds the current
            word.
        _word_idx:
            int - the index of the first letter of the current word in
            _line.
        _NEW_WORDS:
            list[str] - the words to be inserted into line in the place
            of the original word.


        Returns
        -------
        -
            _line: list[str] - the full line in X that held the current
            word with that word removed and the new word(s) inserted in
            the that word's place.

        """

        _word = _line[_word_idx]

        _line.pop(_word_idx)

        # GO THRU _NEW_WORDS BACKWARDS
        for _slot_idx, _new_word in range(len(_NEW_WORDS) - 1, -1, -1):

            _line.insert(_word_idx, _NEW_WORDS[_slot_idx])

            if self.update_lexicon:
                # when prompted to put a word into the lexicon, user can
                # say 'skip always', the word goes into that list, and the
                # user is not prompted again
                if _new_word in self.KNOWN_WORDS or _new_word in self.SKIP_ALWAYS:
                    continue

                # if new word is not KNOWN or not skipped...
                if self.auto_add_to_lexicon:
                    self.LEXICON_ADDENDUM.append(_NEW_WORDS[_slot_idx])
                    self.KNOWN_WORDS.append(_NEW_WORDS[_slot_idx])
                    continue

                print(f"\n*** *{_NEW_WORDS[_slot_idx]}* IS NOT IN LEXICON ***\n")
                _ = self.LexLookupMenu.choose('Select option', allowed='akw')
                if _ == 'a':
                    self.LEXICON_ADDENDUM.append(_NEW_WORDS[_slot_idx])
                    self.KNOWN_WORDS.append(_NEW_WORDS[_slot_idx])
                elif _ == 'k':
                    pass
                elif _ == 'w':
                    self.SKIP_ALWAYS.append(_word)
                else:
                    raise Exception

        del _NEW_WORDS

        return _line


    def transform(
        self,
        X: XContainer,
        copy: Optional[bool] = True
    ):

        """
        Scan tokens in X and prompt for handling of tokens not in the
        Lexicon.


        Parameters
        ----------
        X:
            XContainer - The data in (possibly ragged) 2D array-like
            format.
        copy:
            Optional[bool], default=True - whether to make substitutions
            and deletions directly on X or a deepcopy of X.


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

        # convert X to list-of-lists -- -- -- -- -- -- -- -- -- -- --
        # we know from validation it is legit 2D
        if isinstance(_X, pd.DataFrame):
            _X = list(map(list, _X.values))
        elif isinstance(_X, pl.DataFrame):
            _X = list(map(list, _X.rows()))
        else:
            _X = list(map(list, _X))

        _X: WipXContainer
        # END convert X to list-of-lists -- -- -- -- -- -- -- -- -- --

        # MANAGE THE CONTENTS OF LEXICON ADDENDUM -- -- -- -- -- -- --
        _abort = False

        if self.update_lexicon and len(self.LEXICON_ADDENDUM) != 0:

            print(f'\n*** LEXICON ADDENDUM IS NOT EMPTY ***\n')
            print(f'LEXICON ADDENDUM has {len(self.LEXICON_ADDENDUM)} entries')
            print(f'First 10 in LEXICON ADDENDUM:')
            self._display_lexicon_update(n=10)
            print()

            _opt = vui.validate_user_str(
                f'Empty it(e), Proceed anyway(p), Abort TextLookup(a) > ',
                'AEP'
            )
            if _opt == 'A':
                _abort = True
            elif _opt == 'E':
                self.LEXICON_ADDENDUM = []
            elif _opt == 'P':
                pass
            else:
                raise Exception
            del _opt
        # END MANAGE THE CONTENTS OF LEXICON ADDENDUM -- -- -- -- -- --

        if self.verbose:
            print(f'\nRunning Lexicon cross-reference...')

        _quit = False
        _n_edits = 0
        _word_counter = 0
        for _row_idx in [range(len(_X)) if not _abort else []][0]:

            if self.verbose:
                print(f'\nStarting row {_row_idx+1} of {len(_X)}')
                print(f'\nCurrent state of ')
                self._display_lexicon_update()

            # GO THRU BACKWARDS BECAUSE A SPLIT OR DELETE WILL CHANGE X
            for _word_idx in range(len(_X[_row_idx]) - 1, -1, -1):

                # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
                # Manage in-situ save option if in manual edit mode.
                # the only way that _n_edits can increment is if u get
                # into manual mode, which means both auto_add_to_lexicon
                # and auto_delete are False
                if _n_edits % 10 == 0:
                    _prompt = f'\nSave in-situ changes to file(s) or Continue(c) > '
                    if vui.validate_user_str(_prompt, 'SC') == 'S':
                        _opt = vui.validate_user_str(
                            f'\nSave to csv(c), Save to txt(t), Abort(a)? > ',
                            'CTA'
                        )
                        if _opt == 'C':
                            self.dump_to_csv()
                        elif _opt == 'T':
                            self.dump_to_txt()
                        elif _opt == 'A':
                            pass
                        else:
                            raise Exception
                        del _opt
                    del _prompt
                # END manage in-situ save -- -- -- -- -- -- -- -- -- --

                _word_counter += 1
                if self.verbose and _word_counter % 1000 == 0:
                    print(f'\nWord {_word_counter:,} of {sum(map(len, _X)):,}...')

                _word = _X[_row_idx][_word_idx]

                # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
                # short-circuit for things already known or learned in-situ
                if _word in self.SKIP_ALWAYS:
                    # this may have had words in it from the user at init
                    if self.verbose:
                        print(f'\n*** ALWAYS SKIP *{_word}* ***\n')
                    continue

                if _word in self.DELETE_ALWAYS:
                    # this may have had words in it from the user at init
                    if self.verbose:
                        print(f'\n*** ALWAYS DELETE *{_word}* ***\n')
                    _X[_row_idx].pop(_word_idx)
                    continue

                if _word in self.REPLACE_ALWAYS:
                    # this may have had words in it from the user at init
                    if self.verbose:
                        print(
                            f'\n*** ALWAYS REPLACE *{_word}* WITH '
                            f'*{self.REPLACE_ALWAYS[_word]}* ***\n'
                        )
                    _X[_row_idx] = self._split_or_replace_handler(
                        _X[_row_idx], _word_idx, [self.REPLACE_ALWAYS[_word]]
                    )
                    continue

                if _word in self.SPLIT_ALWAYS:
                    # this may have had words in it from the user at init
                    if self.verbose:
                        print(
                            f'\n*** ALWAYS SPLIT *{_word}* WITH '
                            f'*{"*, *".join(self.SPLIT_ALWAYS[_word])}* ***\n'
                        )
                    _X[_row_idx] = self._split_or_replace_handler(
                        _X[_row_idx], _word_idx, self.SPLIT_ALWAYS[_word]
                    )
                    continue

                # short circuit for numbers
                if self.skip_numbers:
                    try:
                        float(_word)
                        # if get to here its a number, go to next word
                        if self.verbose:
                            print(f'\n*** ALWAYS SKIP NUMBERS *{_word}* ***\n')
                        continue
                    except:
                        pass
                # END short circuit for numbers

                # PUT THIS LAST.... OTHERWISE USER WOULD NEVER BE ABLE
                # TO DELETE, REPLACE, OR SPLIT WORDS ALREADY IN LEXICON
                if _word in self.KNOWN_WORDS:
                    if self.verbose:
                        print(f'\n*** *{_word}* IS ALREADY IN LEXICON ***\n')
                    continue
                # END short-circuit for things already known or learned in-situ
                # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

                # short-circuit for auto-split -- -- -- -- -- -- -- -- --
                # last ditch before auto-add & auto-delete, try to save the word
                # LOOK IF word IS 2 KNOWN WORDS MOOSHED TOGETHER
                # LOOK FOR FIRST VALID SPLIT IF len(word) >= 4
                if self.auto_split and len(_word) >= 4:
                    _NEW_LINE = _auto_word_splitter(
                        _word_idx, _X[_word_idx], self.KNOWN_WORDS, self.verbose
                    )
                    if any(_NEW_LINE):
                        _X[_row_idx] = _NEW_LINE
                        # since auto_word_splitter requires that both halves
                        # already be in the Lexicon, just continue to next word
                        del _NEW_LINE
                        continue
                    # else: if _NEW_LINE is empty, there wasnt a valid split,
                    # just pass, if auto_delete is True, that will delete this
                    # word and go to the next word. if auto_delete is False, it
                    # will also pass thru quasi_auto_splitter, then the user
                    # will have to deal with it manually.
                    del _NEW_LINE
                # END short-circuit for auto-split -- -- -- -- -- -- -- --

                # short-circuit for auto-add -- -- -- -- -- -- -- -- -- --
                # ANYTHING that is in X that is not in Lexicon gets added
                # and stays in X.
                if self.auto_add_to_lexicon:
                    # auto_add_to_lexicon can only be True if update_lexicon=True
                    if self.verbose:
                        print(f'\n*** AUTO-ADD *{_word}* TO LEXICON ADDENDUM ***\n')
                    self.LEXICON_ADDENDUM.append(_word)
                    self.KNOWN_WORDS.append(_word)

                    continue
                # END short-circuit for auto-add -- -- -- -- -- -- -- --

                # short-circuit for auto-delete -- -- -- -- -- -- -- --
                if self.auto_delete:
                    if self.verbose:
                        print(f'\n*** AUTO-DELETE *{_word}* ***\n')
                    _X[_row_idx].pop(_word_idx)
                    continue
                # END short-circuit for auto-delete -- -- -- -- -- -- --


                # v v v MANUAL MODE v v v v v v v v v v v v v v v v v v
                # word is not in KNOWN_WORDS or any repetitive operation holders

                # a manual edit is guaranteed to happen after this point
                _n_edits += 1

                # quasi-automate split recommendation -- -- -- -- -- -- --
                # if we had auto_split=True and we get to here, its because
                # there were no valid splits and just passed thru, so the word
                # will also pass thru here. if auto_split was False and we get
                # to here, we are about to enter manual mode. the user is forced
                # into this as a convenience to partially automate the process
                # of finding splits as opposed to having to manually type 2-way
                # splits over and over.

                if len(_word) >= 4:
                    _NEW_LINE = _quasi_auto_word_splitter(
                        _word_idx, _X[_row_idx], self.KNOWN_WORDS, self.verbose
                    )
                    # if the user did not opt to take any of splits (or if
                    # there werent any), then _NEW_LINE is empty, and the user
                    # is forced into the manual menu.
                    if any(_NEW_LINE):
                        _X[_row_idx] = _NEW_LINE
                        # since quasi_auto_word_splitter requires that both
                        # halves already be in the Lexicon, just continue to
                        # the next word
                        del _NEW_LINE
                        continue

                    del _NEW_LINE
                # END quasi-automate split recommendation -- -- -- -- --

                print(_view_snippet(_X[_row_idx], _word_idx, _span=7))
                print(f"\n*{_word}* IS NOT IN LEXICON\n")
                _opt = self.LexLookupMenu.choose('Select option')

                # manual menu actions -- -- -- -- -- -- -- -- -- -- -- --
                if _opt == 'a':    # 'a': 'Add to Lexicon'
                    # this menu option is not available in LexLookupMenu if
                    # 'update_lexicon' is False
                    self.LEXICON_ADDENDUM.append(_word)
                    self.KNOWN_WORDS.append(_word)
                    if self.verbose:
                        print(f'\n*** ADD *{_word}* TO LEXICON ADDENDUM ***\n')
                    # and X is unchanged
                elif _opt == 'dl':   # 'd': 'Delete', 'l': 'Delete always'
                    _X[_row_idx].pop(_word_idx)
                    if _opt == 'd':
                        if self.verbose:
                            print(f'\n*** ONE-TIME DELETE OF *{_word}* ***\n')
                    elif _opt == 'l':
                        self.DELETE_ALWAYS.append(_word)
                        if self.verbose:
                            print(f'\n*** ALWAYS DELETE *{_word}* ***\n')
                elif _opt in 'ef':   # 'e': 'Replace', 'f': 'Replace always',
                    _new_word = _word_editor(
                        _word,
                        _prompt=f'Enter new word to replace *{_word}*'
                    )
                    _X[_row_idx] = self._split_or_replace_handler(
                        _X[_row_idx], _word_idx, [_new_word]
                    )
                    if _opt == 'e':
                        if self.verbose:
                            print(
                                f'\n*** ONE-TIME REPLACE *{_word}* WITH '
                                f'*{self.REPLACE_ALWAYS[_word]}* ***\n'
                            )
                    elif _opt == 'f':
                        self.REPLACE_ALWAYS[_word] = _new_word
                        if self.verbose:
                            print(
                                f'\n*** ALWAYS REPLACE *{_word}* WITH '
                                f'*{self.REPLACE_ALWAYS[_word]}* ***\n'
                            )
                    del _new_word
                elif _opt in 'kw':   # 'k': 'Skip', 'w': 'Skip always'
                    if _opt == 'k':
                        if self.verbose:
                            print(f'\n*** ONE-TIME SKIP *{_word}* ***\n')
                    elif _opt == 'w':
                        self.SKIP_ALWAYS.append(_word)
                        if self.verbose:
                            print(f'\n*** ALWAYS SKIP *{_word}* ***\n')
                    # a no-op
                    pass
                elif _opt in 'su':   # 's': 'Split', 'u': 'Split always'
                    # this split is different than auto and quasi... those split
                    # on both halves of the original word being in Lexicon, but
                    # here the user might pass something new, so this needs to
                    # run thru _split_or_replace_handler in case update_lexicon
                    # is True and the new words arent in the Lexicon
                    _NEW_WORDS = _manual_word_splitter(
                        _word_idx, _X[_row_idx], self.KNOWN_WORDS, self.verbose
                    )   # cannot be empty
                    _X[_row_idx] = self._split_or_replace_handler(
                        _X[_row_idx],
                        _word_idx,
                        _NEW_WORDS
                    )
                    if _opt == 's':
                        if self.verbose:
                            print(
                                f'\n*** ONE-TIME SPLIT *{_word}* WITH '
                                f'*{"*, *".join(self.SPLIT_ALWAYS[_word])}* ***\n'
                            )
                    elif _opt == 'u':
                        self.SPLIT_ALWAYS[_word] = _NEW_WORDS
                        if self.verbose:
                            print(
                                f'\n*** ALWAYS SPLIT *{_word}* WITH '
                                f'*{"*, *".join(self.SPLIT_ALWAYS[_word])}* ***\n'
                            )
                    del _NEW_WORDS
                elif _opt == 'q':   # 'q': 'Quit'
                    _quit = True
                    break
                else:
                    raise Exception
                # END manual menu actions -- -- -- -- -- -- -- -- -- -- -- --

            if _quit:
                break

        del _n_edits, _word_counter

        if self.verbose:
            print(f'\n*** LEX LOOKUP COMPLETE ***\n')

        if self.update_lexicon and not _abort:
            # show this to the user so they can copy-paste into Lexicon
            if len(self.LEXICON_ADDENDUM) != 0:
                print(f'\n*** COPY AND PASTE THESE WORDS INTO LEXICON ***\n')
                self._display_lexicon_update()

        del _abort


        return _X













