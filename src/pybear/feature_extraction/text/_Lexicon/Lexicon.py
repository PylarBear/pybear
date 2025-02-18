# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Optional, Sequence
from typing_extensions import Union

import os
import glob
import numbers

import numpy as np

from .._TextStatistics.TextStatistics import TextStatistics

from ._methods._add_words import _add_words
from ._methods._check_order import _check_order
from ._methods._delete_words import _delete_words
from ._methods._find_duplicates import _find_duplicates



class Lexicon(TextStatistics):


    def __init__(self) -> None:

        """
        The pybear lexicon of words in the English language. May not be
        exhaustive, though attempts have been made.

        This serves as a list of words in the English language for
        text-cleaning purposes.

        The published pybear lexicon only allows the 26 letters of the
        English alphabet and all must be capitalized. Other characters,
        such as numbers, hyphens, apostrophes, etc., are not allowed.
        For example, entries one may see in the pybear lexicon include
        'APPLE', 'APRICOT', 'APRIL'. Entries that one will not see in
        the published version are 'AREN'T', 'ISN'T' and 'WON'T' (the
        entries would be 'ARENT', 'ISNT', and 'WONT'.) Lexicon has
        validation in place to protect the integrity of the published
        pybear lexicon toward these rules. However, this validation can
        be turned off and local copies can be updated with any strings
        that the user likes.


        Attributes
        ----------
        size_:
            int - the number of words in the pybear English language
            lexicon.
        lexicon_
            list[str] - a list of all the words in the lexicon.
        overall_statistics_:
            dict[str: numbers.Real] - A dictionary that holds information
            about all the words in the Lexicon instance. Available
            statistics are size, uniques count (should be the same as
            size), average word length, standard deviation of word
            length, maximum word length, and minimum word length.
        string_frequency_:
            dict[str, int] - A dictionary that holds the unique words
            in the lexicon and the respective frequency. For the pybear
            lexicon, the frequency of every word should be one.
        startswith_frequency_:
            dict[str, int] - A dictionary that holds the unique first
            characters for all the words in the lexicon (expected to be
            all 26 letters of the English alphabet) and their frequencies
            in the first position. That is, the 'A' key will report the
            number of words in the lexicon that start with 'A'.
        character_frequency_:
            dict[str, int] - A dictionary that holds all the unique
            single characters and their frequencies for all the words in
            the Lexicon instance.
        uniques_
            list[str] - same as lexicon_.


        Notes
        -----
        pybear stores its lexicon in text files that are read from the
        local disk when a Lexicon class is instantiated, populating the
        attributes of the instance. The files are named by the 26 letters
        of the English alphabet, therefore there are 26 files.

        The 'add_words' method allows users who have installed pybear
        locally to add words to their copies of the lexicon. There are
        several validation protocols in place to secure the integrity of
        the published version of the pybear lexicon, and the user must
        consider these when attempting to change their copy of the
        lexicon.

        Again, the published pybear lexicon only consists of the 26
        letters of the English alphabet and must be upper-case. When
        making local additions to the lexicon via the 'add_words' method,
        this validation can be turned off via the 'character_validation',
        'majuscule_validation', and 'file_validation' keyword arguments.
        These allow your lexicon to take non-alpha characters in upper
        or lower case, and allows Lexicon to create new text files for
        itself.


        Examples
        --------
        >>> from pybear.feature_extraction.text import Lexicon
        >>> Lex = Lexicon()
        >>> Lex.size_
        68370
        >>> Lex.lexicon_[:5]
        ['A', 'AA', 'AAA', 'AARDVARK', 'AARDVARKS']
        >>> round(Lex.overall_statistics_['average_length'], 3)
        8.431
        >>> Lex.lookup_string('MONKEY')
        'MONKEY'
        >>> Lex.lookup_string('SUPERCALIFRAGILISTICEXPIALIDOCIOUS')

        >>> Lex.lookup_substring('TCHSTR')
        ['LATCHSTRING', 'LATCHSTRINGS']


        """


        super().__init__()

        self._module_dir = os.path.dirname(os.path.abspath(__file__))
        self._lexicon_dir = os.path.join(self._module_dir, '_lexicon')


        FILES = sorted(glob.glob(os.path.join(self._lexicon_dir, '*.txt')))
        for file in FILES:
            with open(os.path.join(self._lexicon_dir, file)) as f:
                words = np.fromiter(f, dtype='<U40')
                words = np.char.replace(words, '\n', '')
                super().partial_fit(words)

        del FILES, words


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
        the given character substring. Case-sensitive.


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

        return super().lookup_substring(char_seq, case_sensitive=True)


    def lookup_string(self, char_seq: str) -> Union[str, None]:

        """
        Look in the pybear lexicon for an identical word (not substring)
        that exactly matches the given character sequence. If a match is
        found, return that character string. If an exact match is not
        found, return None. Case-sensitive.


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

        return super().lookup_string(char_seq, case_sensitive=True)


    def find_duplicates(self) -> dict[str, numbers.Integral]:

        """
        Find any duplicates in the Lexicon. If any, display to screen
        and return as python dictionary with frequencies.


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


        Return
        ------
        -
            list[str] - vector of any out-of-sequence words in the
            Lexicon.

        """

        return _check_order(self.lexicon_)


    def add_words(
        self,
        WORDS: Union[str, Sequence[str]],
        character_validation: Optional[bool] = True,
        majuscule_validation: Optional[bool] = True,
        file_validation: Optional[bool] = True
    ) -> None:

        """
        Silently update the pybear lexicon text files with the given
        words. Words that are already in the lexicon are silently
        ignored. This is very much a case-sensitive operation.

        The 'validation' parameters allow you to disable the pybear
        lexicon rules. The pybear lexicon does not allow any characters
        that are not one of the 26 letters of the English alphabet.
        Numbers, spaces, and punctuation, for example, are not allowed
        in the formal pybear lexicon. Also, the pybear lexicon requires
        that all entries in the lexicon be MAJUSCULE, i.e., upper-case.
        The published pybear lexicon will always follow these rules.
        When the validation is used it ensures the integrity of the
        lexicon. However, the user can override this validation for
        local copies of pybear by setting 'character_validation',
        'majuscule_validation', and / or 'file_validation' to False. If
        you want your lexicon to have strings that contain numbers,
        spaces, punctuation, and have different cases, then set the
        validation to False and add your strings to the lexicon via this
        method.

        pybear stores words in the lexicon text files based on the first
        character of the string. So a word like 'APPLE' is stored in a
        file named 'lexicon_A' (this is the default pybear way.) A word
        like 'apple' would be stored in a file named 'lexicon_a'. Keep
        in mind that the pybear lexicon is built with all capitalized
        words and file names and these are the only ones that exist out
        of the box. If you were to turn off the 'majuscule_validation'
        and 'file_validation' and pass the word 'apple' to this method,
        it will NOT append 'APPLE' to the 'lexicon_A' file, a new lexicon
        file called 'lexicon_a' will be created and the word 'apple'
        will be put into it.

        The Lexicon instance reloads the lexicon from disk and refills
        the attributes when update is complete.


        Parameters
        ----------
        WORDS:
            Union[str, Sequence[str]] - the word or words to be added to
            the pybear lexicon. Cannot be an empty string or an empty
            sequence. Words that are already in the lexicon are silently
            ignored.
        character_validation:
            Optional[bool], default = True - whether to apply pybear
            lexicon character validation to the word or sequence of
            words. pybear lexicon allows only the 26 letters in the
            English language, no others. No spaces, no hyphens, no
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
        file_validation:
            Optional[bool], default = True - whether to apply pybear
            lexicon file name validation to the word or sequence of
            words. The formal pybear lexicon only allows words to start
            with the 26 upper-case letters of the English alphabet (which
            then dictates the file name in which it will be stored). If
            True, any disallowed characters in the first position will
            raise an exception during validation. If False, any character is
            accepted, which may then necessitate that a file be created.


        Return
        ------
        -
            None

        """


        _add_words(
            WORDS,
            self._lexicon_dir,
            character_validation=character_validation,
            majuscule_validation=majuscule_validation,
            file_validation=file_validation
        )

        # _add_words writes new words to files. need to re-read files
        # into the instance and rebuild the lexicon and attributes.
        self.__init__()


    def delete_words(
        self,
        WORDS: Union[str, Sequence[str]]
    ):

        """
        Remove the given word(s) from the pybear lexicon text files.
        Case sensitive! Any words that are not in the pybear lexicon are
        silently ignored.


        Parameters
        ----------
        WORDS:
            Union[str, Sequence[str]] - the word or words to remove from
            the pybear lexicon. Cannot be an empty string or an empty
            sequence.


        Return
        ------
        -
            None

        """


        _delete_words(
            WORDS,
            self._lexicon_dir
        )

        # _delete_words removes words from the files. need to re-read
        # files into the instance and rebuild the attributes.
        self.__init__()








