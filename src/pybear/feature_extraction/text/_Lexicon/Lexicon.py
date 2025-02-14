# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#
import numbers

import numpy.typing as npt

import os

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
            NDArray[str] - the words in the pybear English language
            lexicon.
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


        Notes
        -----
        pizza


        Examples
        --------
        pizza



        """


        super().__init__()

        module_dir = os.path.dirname(os.path.abspath(__file__))
        lexicon_dir = os.path.join(module_dir, '_lexicon')
        FILES = [f'lexicon_{_}' for _ in 'abcdefghijklmnopqrstuvwxyz']
        for file in FILES:
            with open(os.path.join(lexicon_dir, file + f'.txt')) as f:
                # pizza can this generator be changed uce?
                words = np.fromiter((_ for _ in f), dtype='<U40')
                words = np.char.replace(words, '\n', '')
                super().partial_fit(words)
                # pizza
                # WORDS = np.hstack((WORDS, words), dtype='<U40')


        self.lexicon_ = np.array(self.uniques_, dtype='<U30')

        del module_dir, lexicon_dir, FILES, words


    @TextStatistics.uniques_.getter
    def uniques_(self):
        raise AttributeError(f"'uniques_' is blocked, use 'lexicon_'")


    @TextStatistics.uniques_.setter
    def uniques_(self, value):
        raise AttributeError(f"'uniques_' is blocked, use 'lexicon_'")


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


    def find_duplicates(self) -> dict[str, numbers.Integral]:

        """
        Find any duplicates in the Lexicon. If any, display to screen
        and return as numpy vector.


        Parameters
        ----------
        None


        Return
        ------
        -
            Duplicates: dict[str, numbers.Integral] - vector of any
            duplicates in the pybear lexicon and their frequencies.

        """

        return _find_duplicates(self.string_frequency_)


    def check_order(self) -> npt.NDArray[str]:

        """
        Determine if words stored in the Lexicon files are out of
        alphabetical order by comparing the words as stored against a
        sorted vector of the words. Displays any out-of-order words to
        screen and returns a numpy vector of the words.


        Parameters
        ----------
        None


        Return
        ------
        -
            OUT_OF_ORDER: NDArray[str] - vector of any out of sequence
            words in the Lexicon.

        """

        return _check_order(self.lexicon_)


    def add_words(
        self,
        # pizza one word at a time or a list?
    ):

        """
        Add words to the lexicon text files.
        Pizza add more stuff.


        Parameters
        ----------


        Return
        ------
        -
            None

        """

        _add_words(self.lexicon_)

        # pizza
        # _add_words writes new words to files. need to re-read files into
        # the instance and rebuild the attributes.
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


    # pizza once this is in TextStatistics it can come out of here
    # def lookup_substring(
    #         self,
    #         char_seq: str,
    #         *,
    #         bypass_validation=False
    #     ) -> np.ndarray:
    #
    #     """
    #     Return a numpy array of all words in the Lexicon that contain the
    #     given character string.
    #
    #     Parameters
    #     ----------
    #     char_seq:
    #         str - alpha character string to be looked up
    #     bypass_validation:
    #         bool - if True, bypass _validation of char_seq
    #
    #     Return
    #     ------
    #     -
    #         SELECTED_WORDS: np.ndarray - list of all words in the Lexicon
    #         that contain the given character string
    #
    #     """
    #
    #     bypass_validation = arg_kwarg_validater(
    #         bypass_validation,
    #         'bypass_validation',
    #         [True, False, None],
    #         get_module_name(str(sys.modules[__name__])),
    #         inspect.stack()[0][3],
    #         return_if_none=True
    #     )
    #
    #     if not bypass_validation:
    #         char_seq = self._string_validation(char_seq)
    #
    #     MASK = np.fromiter(
    #         map(lambda x: x.find(char_seq, 0, len(char_seq)) + 1, self.LEXICON),
    #         dtype=bool
    #     )
    #     SELECTED_WORDS = self.lexicon()[MASK]
    #     del char_seq, MASK
    #
    #     return SELECTED_WORDS


    # pizza once this is in TextStatistics it can come out of here
    # def lookup_word(
    #         self,
    #         char_seq: str,
    #         *,
    #         bypass_validation: bool=False
    #     ):
    #
    #     """
    #     Return a boolean indicating if a given character string matches
    #     a word in the Lexicon.
    #
    #     Parameters
    #     ----------
    #     char_seq:
    #         str - alpha character string to be looked up
    #     bypass_validation:
    #         bool - if True, bypass _validation of char_seq
    #
    #     Return
    #     ------
    #     -
    #         bool: boolean indicating if the given character string matches
    #             a word in the Lexicon.
    #
    #     """
    #
    #     bypass_validation = arg_kwarg_validater(
    #         bypass_validation,
    #         'bypass_validation',
    #         [True, False, None],
    #         get_module_name(str(sys.modules[__name__])),
    #         inspect.stack()[0][3],
    #         return_if_none=True
    #     )
    #
    #     if not bypass_validation:
    #         char_seq = self._string_validation(char_seq)
    #
    #     return char_seq in self.LEXICON


    # pizza this probably comes out!
    # def statistics(self):
    #
    #     """
    #     Print statistics about the Lexicon to the screen. Returns nothing.
    #     Statistics reported include
    #     - size
    #     - uniques count
    #     - average length and standard deviation
    #     - max word length
    #     - min word length
    #     - 'starts with' frequency
    #     - letter frequency
    #     - top word frequencies,
    #     - top longest words
    #
    #     Parameters
    #     ----------
    #     None
    #
    #     Return
    #     ------
    #     -
    #         None
    #
    #     """
    #
    #
    #     _statistics._statistics(self.LEXICON)


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













