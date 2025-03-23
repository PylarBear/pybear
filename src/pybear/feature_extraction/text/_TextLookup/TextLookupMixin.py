# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Optional, Sequence
from typing_extensions import Self, Union

import numbers

from ._shared._type_aliases import XContainer

from ....utilities._DictMenuPrint import DictMenuPrint

from ....base import (
    FileDumpMixin,
    FitTransformMixin,
    GetParamsMixin,
    ReprMixin,
    SetParamsMixin,
    check_is_fitted
)



class TextLookupMixin(
    FileDumpMixin,
    FitTransformMixin,
    GetParamsMixin,
    ReprMixin,
    SetParamsMixin
):

    """
    A mixin for TextLookup and TextLookupRealTime that provides everything
    except docs, partial_fit, and transform.

    """

    def __init__(
        self,
        *,
        update_lexicon: Optional[bool] = False,
        skip_numbers: Optional[bool] = True,
        auto_split: Optional[bool] = True,
        auto_add_to_lexicon: Optional[bool] = False,
        auto_delete: Optional[bool] = False,
        DELETE_ALWAYS: Optional[Union[Sequence[str], None]] = None,
        REPLACE_ALWAYS: Optional[Union[dict[str, str], None]] = None,
        SKIP_ALWAYS: Optional[Union[Sequence[str], None]] = None,
        SPLIT_ALWAYS: Optional[Union[dict[str, Sequence[str]], None]] = None,
        remove_empty_rows: Optional[bool] = False,
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
        self.remove_empty_rows = remove_empty_rows
        self.verbose = verbose

        # needs to get self._LEX_LOOK_DICT from the child
        if not self.update_lexicon and 'a' in self._LEX_LOOK_DICT:
            del self._LEX_LOOK_DICT['a']

        self._LexLookupMenu = DictMenuPrint(
            self._LEX_LOOK_DICT,
            disp_width=75,
            fixed_col_width=25
        )
    # END init ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **


    def reset(self) -> Self:
        """
        Reset the TextLookup instance. This will remove all attributes
        that are exposed during transform.


        Returns
        -------
        -
            None.

        """

        if hasattr(self, 'n_rows_'):
            delattr(self, 'n_rows_')
        if hasattr(self, 'row_support_'):
            delattr(self, 'row_support_')

        if hasattr(self, 'DELETE_ALWAYS_'):
            delattr(self, 'DELETE_ALWAYS_')
        if hasattr(self, 'REPLACE_ALWAYS_'):
            delattr(self, 'REPLACE_ALWAYS_')
        if hasattr(self, 'SKIP_ALWAYS_'):
            delattr(self, 'SKIP_ALWAYS_')
        if hasattr(self, 'SPLIT_ALWAYS_'):
            delattr(self, 'SPLIT_ALWAYS_')

        if hasattr(self, 'LEXICON_ADDENDUM_'):
            delattr(self, 'LEXICON_ADDENDUM_')
        if hasattr(self, 'KNOWN_WORDS_'):
            delattr(self, 'KNOWN_WORDS_')
        if hasattr(self, '_OOV'):
            delattr(self, '_OOV')

        return self


    def get_metadata_routing(self):
        raise NotImplementedError(
            f"'get_metadata_routing' is not implemented in TextLookup"
        )


    # def get_params
    # handled by GetParamsMixin


    # def set_params
    # handled by SetParamsMixin


    # def fit_transform
    # handled by FitTransformMixin


    def score(
        self,
        X: XContainer,
        y: Optional[Union[any, None]] = None
    ) -> None:

        """
        No-op score method. Needs to be here for dask_ml wrappers.


        Parameters
        ----------
        X:
            XContainer - the (possibly ragged) 2D container of text to
            have its contents cross-referenced against the pybear
            Lexicon. Ignored.
        y:
            Optional[Union[any, None]], default=None - the target for
            the data. Always ignored.


        Return
        ------
        -
            None


        """

        check_is_fitted(self)

        return


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
        if len(self.LEXICON_ADDENDUM_) == 0:
            print(f'*** EMPTY ***')
        else:
            self.LEXICON_ADDENDUM_.sort()
            print(f'[')
            for _ in self.LEXICON_ADDENDUM_[:(n or len(self.LEXICON_ADDENDUM_))]:
                print(f'    "{_}"{"" if _ == self.LEXICON_ADDENDUM_[-1] else ","}')
            print(f']')
            print()


    def _split_or_replace_handler(
        self,
        _line: list[str],
        _word_idx: numbers.Integral,
        _NEW_WORDS: list[str]
    ) -> list[str]:

        """
        Handle removing a user-identified word from a line, substituting
        in new word(s), and updating the LEXICON_ADDENDUM, if applicable.

        This is called after split, split always, replace, and replace
        always.


        Parameters
        ----------
        _line:
            list[str] - the full line of the data that holds the current
            word.
        _word_idx:
            int - the index of the current word in _line.
        _NEW_WORDS:
            list[str] - the word(s) to be inserted into _line in place
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
        for _slot_idx in range(len(_NEW_WORDS) - 1, -1, -1):

            _new_word = _NEW_WORDS[_slot_idx]

            _line.insert(_word_idx, _new_word)

            if self.update_lexicon:
                # when prompted to put a word into the lexicon, user can
                # say 'skip always', the word goes into that list, and the
                # user is not prompted again
                # conveniently for plain TextLookup, when in (partial_)fit and
                # the user picks one of the 2 options 'a' or 'w', it causes
                # the word to go in one of the lists which forces bypass
                # here in transform and avoids the menu.
                if _new_word in self.KNOWN_WORDS_ \
                        or _new_word in self.SKIP_ALWAYS_:
                    continue

                # if new word is not KNOWN or not skipped...
                if self.auto_add_to_lexicon:
                    self.LEXICON_ADDENDUM_.append(_NEW_WORDS[_slot_idx])
                    self.KNOWN_WORDS_.insert(0, _NEW_WORDS[_slot_idx])
                    continue

                print(f"\n*** *{_NEW_WORDS[_slot_idx]}* IS NOT IN LEXICON ***\n")
                _ = self._LexLookupMenu.choose('Select option', allowed='aw')
                if _ == 'a':
                    self.LEXICON_ADDENDUM_.append(_NEW_WORDS[_slot_idx])
                    self.KNOWN_WORDS_.insert(0, _NEW_WORDS[_slot_idx])
                elif _ == 'w':
                    self.SKIP_ALWAYS_.append(_word)
                else:
                    raise Exception

        del _NEW_WORDS

        return _line













