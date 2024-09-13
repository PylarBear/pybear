# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import numpy as np
import os, sys, inspect
from ...text import (
    alphanumeric_str as ans,
    _statistics as _statistics
)
from ....utilities._get_module_name import get_module_name
from ....data_validation import arg_kwarg_validater
from ...text._Lexicon._old_py_lexicon import (
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


# size
# find_duplicates
# check_order
# _string_validation
# lookup_substring
# lookup_word
# statistics
# lexicon
# _old_py_lexicon


class Lexicon:

    """
    The lexicon of words in the the English vocabulary. May not be
    exhaustive.


    """


    def __init__(self):

        self.LEXICON = self.lexicon()


    def size(self) -> int:

        """
        The count of words in the Lexicon.

        Parameters
        ----------
        None

        Return
        ------
        -
            len(Lexicon): int

        """

        return len(self.LEXICON)


    def find_duplicates(self) -> np.ndarray:

        """
        Find any duplicates in the Lexicon. If any, display to screen
        and return as numpy vector.

        Parameters
        ----------
        None

        Return
        ------
        -
            Duplicates: np.ndarray - vector of any duplicates in the
            Lexicon.

        """

        UNIQUES, COUNTS = np.unique(self.LEXICON, return_counts=True)
        if len(UNIQUES) == len(self.LEXICON):
            print(f'\n*** THERE ARE NO DUPLICATES IN THE LEXICON ***\n')
            del UNIQUES, COUNTS
            return np.empty(0, dtype='<U40')

        else:
            MASTER_SORT = np.flip(np.argsort(COUNTS))
            MASK = MASTER_SORT[..., COUNTS[..., MASTER_SORT] > 1]
            MASKED_SORTED_UNIQUES = UNIQUES[..., MASK]
            MASKED_SORTED_COUNTS = COUNTS[..., MASK]
            INDICES = np.unique(MASKED_SORTED_UNIQUES, return_index=True)[1]
            DUPLICATES = MASKED_SORTED_UNIQUES[INDICES]
            COUNTS = MASKED_SORTED_COUNTS[INDICES]

            del MASTER_SORT, UNIQUES, MASK
            del MASKED_SORTED_UNIQUES, MASKED_SORTED_COUNTS, INDICES

            if len(DUPLICATES) == 0:
                print(f'\n*** THERE ARE NO DUPLICATED IN LEXICON ***\n')
            else:
                print()
                print(f'*' * 79)
                print(f'\n DUPLICATE'.ljust(30) + f'COUNT')
                print(f'-' * 40)
                [print(f'{d}'.ljust(30) + f'{c}') for d,c in zip(DUPLICATES, COUNTS)]
                print()
                print(f'*' * 79)

            del COUNTS

            return DUPLICATES


    def check_order(self) -> np.ndarray:

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
            OUT_OF_ORDER: np.ndarray - vector of any duplicates in the
            Lexicon.

        """

        __ = np.unique(self.LEXICON)

        if np.array_equiv(self.LEXICON, __):
            print(f'\n*** LEXICON IS IN ALPHABETICAL ORDER ***\n')
            return np.empty(0, dtype='<U40')

        else:
            OUT_OF_ORDER = []
            for idx in range(len(__)):
                if self.LEXICON[idx] != __[idx]:
                    OUT_OF_ORDER.append(__[idx])
            if len(OUT_OF_ORDER) > 0:
                print(f'OUT OF ORDER:')
                print(OUT_OF_ORDER)

            return np.array(OUT_OF_ORDER, dtype='<U40')


    def _string_validation(self, char_seq:str) -> str:

        """Validate alpha character string entry and return in all caps.

        Parameters
        ----------
        char_seq:
            str - alpha character string to be validated

        Return
        ------
        -
            char_seq: str - Validated alpha character string.

        """

        err_msg = f'char_seq MUST BE A str OF alpha characters'

        if not isinstance(char_seq, str):
            raise TypeError(err_msg)

        for _ in char_seq:
            if _.upper() not in ans.alphabet_str_upper():
                raise ValueError(err_msg)

        return char_seq.upper()


    def lookup_substring(
            self,
            char_seq: str,
            *,
            bypass_validation=False
        ) -> np.ndarray:

        """
        Return a numpy array of all words in the Lexicon that contain the
        given character string.

        Parameters
        ----------
        char_seq:
            str - alpha character string to be looked up
        bypass_validation:
            bool - if True, bypass _validation of char_seq

        Return
        ------
        -
            SELECTED_WORDS: np.ndarray - list of all words in the Lexicon
            that contain the given character string

        """

        bypass_validation = arg_kwarg_validater(
            bypass_validation,
            'bypass_validation',
            [True, False, None],
            get_module_name(str(sys.modules[__name__])),
            inspect.stack()[0][3],
            return_if_none=True
        )

        if not bypass_validation:
            char_seq = self._string_validation(char_seq)

        MASK = np.fromiter(
            map(lambda x: x.find(char_seq, 0, len(char_seq)) + 1, self.LEXICON),
            dtype=bool
        )
        SELECTED_WORDS = self.lexicon()[MASK]
        del char_seq, MASK

        return SELECTED_WORDS


    def lookup_word(
            self,
            char_seq: str,
            *,
            bypass_validation: bool=False
        ):

        """
        Return a boolean indicating if a given character string matches
        a word in the Lexicon.

        Parameters
        ----------
        char_seq:
            str - alpha character string to be looked up
        bypass_validation:
            bool - if True, bypass _validation of char_seq

        Return
        ------
        -
            bool: boolean indicating if the given character string matches
                a word in the Lexicon.

        """

        bypass_validation = arg_kwarg_validater(
            bypass_validation,
            'bypass_validation',
            [True, False, None],
            get_module_name(str(sys.modules[__name__])),
            inspect.stack()[0][3],
            return_if_none=True
        )

        if not bypass_validation:
            char_seq = self._string_validation(char_seq)

        return char_seq in self.LEXICON


    def statistics(self):

        """
        Print statistics about the Lexicon to the screen. Returns nothing.
        Statistics reported include
        - size
        - uniques count
        - average length and standard deviation
        - max word length
        - min word length
        - 'starts with' frequency
        - letter frequency
        - top word frequencies,
        - top longest words

        Parameters
        ----------
        None

        Return
        ------
        -
            None

        """


        _statistics._statistics(self.LEXICON)


    def lexicon(self) -> np.ndarray:

        """
        Generate the Lexicon as a numpy vector from files.

        Parameters
        ----------
        None

        Return
        ------
        -
            WORDS: np.ndarray - the full alphabetically sorted Lexicon


        """

        module_dir = os.path.dirname(os.path.abspath(__file__))
        lexicon_dir = os.path.join(module_dir, '_lexicon')
        FILES = [f'lexicon_{_}' for _ in 'abcdefghijklmnopqrstuvwxyz']
        WORDS = np.empty(0, dtype='<U40')
        for file in FILES:

            with open(os.path.join(lexicon_dir, file + f'.txt')) as f:
                words = np.fromiter((_ for _ in f), dtype='<U40')
                words = np.char.replace(words, '\n', '')
                WORDS = np.insert(WORDS, len(WORDS), words, axis=0)

        return WORDS


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



if __name__ == '__main__':

    Lexicon().check_order()
    Lexicon().find_duplicates()
    print(Lexicon().size())
    Lexicon().statistics()








