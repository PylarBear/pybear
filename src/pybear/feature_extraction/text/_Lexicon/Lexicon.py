# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from typing import Optional, Sequence
from typing_extensions import Union

import os
import numbers

import numpy as np

from .._TextStatistics.TextStatistics import TextStatistics

from ._methods._add_words import _add_words
from ._methods._check_order import _check_order
from ._methods._delete_words import _delete_words
from ._methods._find_duplicates import _find_duplicates

from ._old_py_lexicon import (
    lexicon_a as la,
    lexicon_b as lb,
    lexicon_c as lc,
    lexicon_d as ld,
    lexicon_e as le,
    lexicon_f as lf,
    lexicon_g as lg,
    lexicon_h as lh,
    lexicon_i as li,
    lexicon_j as lj,
    lexicon_k as lk,
    lexicon_l as ll,
    lexicon_m as lm,
    lexicon_n as ln,
    lexicon_o as lo,
    lexicon_p as lp,
    lexicon_q as lq,
    lexicon_r as lr,
    lexicon_sa_sm as lsa,
    lexicon_sn_sz as lsn,
    lexicon_t as lt,
    lexicon_u as lu,
    lexicon_v as lv,
    lexicon_w as lw,
    lexicon_x as lx,
    lexicon_y as ly,
    lexicon_z as lz
)



class Lexicon(TextStatistics):

    # size
    # find_duplicates
    # check_order
    # _string_validation
    # statistics
    # lexicon
    # _old_py_lexicon

    # from TextStatistics:
    # @property ------
    # size_
    # uniques_

    # methods ------
    # _reset
    # get_params            --- pizza probably should block
    # partial_fit           --- pizza probably should block
    # fit                   --- pizza probably should block
    # print_overall_statistics
    # print_startswith_frequency
    # print_character_frequency
    # print_string_frequency
    # get_longest_strings
    # print_longest_strings
    # get_shortest_strings
    # print_shortest_strings
    # lookup substring
    # lookup string
    # score                 --- pizza probably should block



    def __init__(self) -> None:

        """
        The pybear lexicon of words in the English language. May not be
        exhaustive.

        This serves as a list of legitimate words in the English language.


        Attributes
        ----------
        size_:
            int - the number of words in the pybear lexicon.
        lexicon_
            list[str] - the words in the pybear English language lexicon.
        overall_statistics_:
            dict[str: numbers.Real] - A dictionary that holds information
            about all the strings fitted on the TextStatistics instance.
            Available statistics are size (number of strings seen during
            fitting), uniques count, average string length, standara
            deviation of string length, maximum string length, and
            minimum string length.
        string_frequency_:
            dict[str, int] - A dictionary that holds the unique strings
            and the respective number of occurrences seen during fitting.
        startswith_frequency_:
            dict[str, int] - A dictionary that holds the first characters
            and their frequencies in the first position for all the
            strings fitted on the TextStatistics instance.
        character_frequency_:
            dict[str, int] - A dictionary that holds all the unique
            single characters and their frequencies for all the strings
            fitted on the TextStatistics instance.
        uniques_
            list[str] - same as lexicon_.


        Notes
        -----
        pizza


        Examples
        --------
        pizza



        """


        super().__init__()

        self._module_dir = os.path.dirname(os.path.abspath(__file__))
        self._lexicon_dir = os.path.join(self._module_dir, '_lexicon')


        FILES = [f'lexicon_{_}' for _ in 'abcdefghijklmnopqrstuvwxyz']
        for file in FILES:
            with open(os.path.join(self._lexicon_dir, file + f'.txt')) as f:
                # pizza can this generator be changed uce?
                # (_ for _ in f)
                words = np.fromiter(f, dtype='<U40')
                words = np.char.replace(words, '\n', '')
                super().partial_fit(words)
                # pizza
                # WORDS = np.hstack((WORDS, words), dtype='<U40')

        del FILES, words


    # pizza
    # @TextStatistics.uniques_.getter
    # def uniques_(self):
    #     raise AttributeError(
    #         f"'uniques_' attribute is blocked, use 'lexicon_'"
    #     )

    @property
    def lexicon_(self):
        return self.uniques_


    def _reset(self):
        raise AttributeError(f"'_reset' is blocked")


    def get_params(self):
        raise AttributeError(f"'get_params' is blocked")


    def partial_fit(self):
        raise AttributeError(f"'partial_fit' is blocked")


    def fit(self):
        raise AttributeError(f"'fit' is blocked")


    def score(self):
        raise AttributeError(f"'score' is blocked")


    def lookup_substring(self, char_seq: str) -> list[str]:

        """
        Return a list of all words in the pybear lexicon that contain
        the given character substring.


        Parameters
        ----------
        char_seq:
            str - character substring to be looked up against the words
            in the pybear lexicon.


        Return
        ------
        -
            list[str] - list of all words in the pybear lexicon that
            contain the given character substring. Returns an empty list
            if there are no matches.


        """

        if not isinstance(char_seq, str):
            raise TypeError(f"'char_seq' must be a string")

        return super().lookup_substring(char_seq.upper(), case_sensitive=True)


    def lookup_string(self, char_seq: str) -> Union[str, None]:

        """
        Look in the pybear lexicon for an identical word (not substring)
        that exactly matches the given character sequence. If a match is
        found, return that character string. If an exact match is not
        found, return None.


        Parameters
        ----------
        char_seq:
            str - character string to be looked up against the words in
            the pybear lexicon.


        Return
        ------
        -
            Union[str, None] - if 'char_seq' is in the pybear lexicon,
            return the word; if there is no match, return None.


        """

        if not isinstance(char_seq, str):
            raise TypeError(f"'char_seq' must be a string")

        return super().lookup_string(char_seq.upper(), case_sensitive=True)


    def find_duplicates(self) -> dict[str, numbers.Integral]:

        """
        Find any duplicates in the Lexicon. If any, display to screen
        and return as python dictionary with frequencies.


        Parameters
        ----------
        None


        Return
        ------
        -
            Duplicates: dict[str, numbers.Integral] - any duplicates in
            the pybear lexicon and their frequencies.

        """

        return _find_duplicates(self.string_frequency_)


    def check_order(self) -> list[str]:

        """
        Determine if words stored in the Lexicon files are out of
        alphabetical order by comparing the words as stored against a
        sorted list of the words. Displays any out-of-order words to
        screen and returns a python list of the words.


        Parameters
        ----------
        None


        Return
        ------
        -
            list[str] - vector of any out of sequence words in the
            Lexicon.

        """

        return _check_order(self.lexicon_)


    def add_words(
        self,
        WORDS: Union[str, Sequence[str]],
        character_validation: Optional[bool] = True,
        majuscule_validation: Optional[bool] = True
    ):

        """
        Silently update the pybear lexicon text files with the given
        words. Words that are already in the lexicon are skipped.

        The 'validation' parameters allow you to disable the pybear
        lexicon rules. The pybear lexicon does not allow any characters
        that are not one of the 26 letters of the English alphabet.
        Numbers, spaces, and punctuation, for example, are not allowed
        in the formal pybear lexicon. Also, the pybear lexicon requires
        that all entries in the lexicon be MAJUSCULE, i.e., upper-case.
        The installed pybear lexicon will always follow these rules.
        When the validation is used it ensures the integrity of the
        lexicon. However, the user can override this validation for
        local copies of pybear by setting 'character_validation' and
        'majuscule_validation' to False. If you want your lexicon to
        have strings that contain numbers, spaces, punctuation, and have
        different cases, then set the validation to False and add your
        strings to the lexicon via this method. There is a caveat,
        however, in that whatever is to be added to the lexicon via this
        method must start with one of the 26 letters of the English
        alphabet. 'over-easy' and 'python2@gmail.com' could be added to
        the lexicon, but '2' and '@gmail' are not accepted.

        The Lexicon instance reloads the lexicon from disk and refills
        the attributes when update is complete.


        Parameters
        ----------
        WORDS:
            Union[str, Sequence[str]] - the word or words to be appended
            to the pybear lexicon. Cannot be an empty string or an empty
            sequence.
        character_validation:
            Optional[bool], default = True - whether to apply pybear
            lexicon character validation to the word or sequence of
            words. pybear lexicon allows only the 26 letters in the
            English language, no others. No spaces, no hypens, no
            apostrophes. If True, any non-alpha characters will raise
            an exception during validation. If False, any string
            character is accepted.
        majuscule_validation:
            Optional[bool], default = True - whether to apply pybear
            lexicon majuscule validation to the word or sequence of
            words. The pybear lexicon requires all characters be
            majuscule, i.e., EVERYTHING MUST BE UPPER-CASE. If True,
            any non-majuscule characters will raise an exception during
            validation. If False, any case is accepted.


        Return
        ------
        -
            None

        """

        _add_words(
            WORDS,
            self._lexicon_dir,
            character_validation=character_validation,
            majuscule_validation=majuscule_validation
        )

        # _add_words writes new words to files. need to re-read files
        # into the instance and rebuild the lexicon and attributes.
        self.__init__()


    def delete_words(
        self,
        # pizza one word at a time or a list?
    ):
        """
        Delete words from the lexicon text files.
        Pizza add more stuff.


        Parameters
        ----------


        Return
        ------
        -
            None

        """


        _delete_words()

        # pizza
        # pizza
        # _delete_words removes words from the files. need to re-read files
        # into the instance and rebuild the attributes.
        self.__init__()


    # pizza this can probably come out uce!
    # def _string_validation(self, char_seq:str) -> str:
    #
    #     """
    #     Validate alpha character string entry and return in all caps.
    #
    #
    #     Parameters
    #     ----------
    #     char_seq:
    #         str - alpha character string to be validated
    #
    #
    #     Return
    #     ------
    #     -
    #         char_seq: str - Validated alpha character string.
    #
    #     """
    #
    #     err_msg = f'char_seq MUST BE A str OF alpha characters'
    #
    #     if not isinstance(char_seq, str):
    #         raise TypeError(err_msg)
    #
    #     for _ in char_seq:
    #         if _.upper() not in ans.alphabet_str_upper():
    #             raise ValueError(err_msg)
    #
    #     return char_seq.upper()


    def _old_py_lexicon(self):

        """
        The original Lexicon storage format, as importable vectors of
        words. Superseded by text files.


        """

        return np.hstack((
            la.lexicon_a(),
            lb.lexicon_b(),
            lc.lexicon_c(),
            ld.lexicon_d(),
            le.lexicon_e(),
            lf.lexicon_f(),
            lg.lexicon_g(),
            lh.lexicon_h(),
            li.lexicon_i(),
            lj.lexicon_j(),
            lk.lexicon_k(),
            ll.lexicon_l(),
            lm.lexicon_m(),
            ln.lexicon_n(),
            lo.lexicon_o(),
            lp.lexicon_p(),
            lq.lexicon_q(),
            lr.lexicon_r(),
            lsa.lexicon_sa_sm(),
            lsn.lexicon_sn_sz(),
            lt.lexicon_t(),
            lu.lexicon_u(),
            lv.lexicon_v(),
            lw.lexicon_w(),
            lx.lexicon_x(),
            ly.lexicon_y(),
            lz.lexicon_z()
        ))













