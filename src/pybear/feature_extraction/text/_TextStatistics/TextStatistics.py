# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import (
    Optional,
    Sequence,
)
from typing_extensions import Self
from ._type_aliases import (
    OverallStatisticsType,
    StartsWithFrequencyType,
    CharacterFrequencyType,
    StringFrequencyType,
    LongestStringsType,
    ShortestStringsType
)
import numpy.typing as npt

import numbers

import numpy as np

from ._validation._strings import _val_strings
from ._validation._overall_statistics import _val_overall_statistics
from ._validation._uniques import _val_uniques
from ._validation._string_frequency import _val_string_frequency
from ._validation._startswith_frequency import _val_startswith_frequency
from ._validation._character_frequency import _val_character_frequency

from ._partial_fit._build_overall_statistics import _build_overall_statistics
from ._partial_fit._merge_overall_statistics import _merge_overall_statistics
from ._partial_fit._build_string_frequency import _build_string_frequency
from ._partial_fit._merge_string_frequency import _merge_string_frequency
from ._partial_fit._build_startswith_frequency import _build_startswith_frequency
from ._partial_fit._merge_startswith_frequency import _merge_startswith_frequency
from ._partial_fit._build_character_frequency import _build_character_frequency
from ._partial_fit._merge_character_frequency import _merge_character_frequency

from ._print._overall_statistics import _print_overall_statistics
from ._print._startswith_frequency import _print_starts_with_frequency
from ._print._string_frequency import _print_string_frequency
from ._print._character_frequency import _print_character_frequency
from ._print._longest_strings import _print_longest_strings
from ._print._shortest_strings import _print_shortest_strings

from ._get._get_longest_strings import _get_longest_strings
from ._get._get_shortest_strings import _get_shortest_strings

from ._lookup._lookup_substring import _lookup_substring
from ._lookup._lookup_string import _lookup_string

from ....base import (
    ReprMixin,
    check_is_fitted
)



class TextStatistics(ReprMixin):

    """
    Generate summary information about a list or multiple lists of
    strings. Statistics include:

    - size (number of strings fitted)

    - unique strings count

    - average length and standard deviation of all strings

    - max string length

    - min string length

    - string frequencies

    - 'starts with' frequency

    - single character frequency

    - longest strings

    - shortest strings

    TextStatistics has 2 scikit-style methods, partial_fit and fit. It
    does not have a transform method, and because the instance does not
    take parameters, it does not have a set_params method. TextStatistics
    does have other methods that allow access to certain functionality,
    such as conveniently printing summary information from attributes to
    screen. See the methods section of the docs.

    TextStatistics can be fit on a single batch of data via :method: fit,
    and can be fit in batches via :method: partial_fit. The fit method
    resets the instance with each call, that is, all information held
    within the instance prior is deleted and the new fit information
    repopulates. The partial_fit method, however, does not reset and
    accumulates information across all batches seen. This makes
    TextStatistics suitable for streaming data and batch-wise training,
    such as with dask_ml Incremental and ParallelPostFit wrappers.

    TextStatistics accepts 1D list-likes containing only strings. This
    includes numpy arrays, python lists, sets, and tuples, and pandas
    Series.


    Attributes
    ----------
    size_:
        numbers.Integral - The number of strings fitted on the
        TextStatistics instance.
    uniques_:
        Sequence[str] - A 1D sequence of the unique strings fitted
        on the TextStatistics instance.
    overall_statistics_:
        dict[str: numbers.Real] - A dictionary that holds information
        about all the strings fitted on the TextStatistics instance,
        such as average string length, maximum string length, total
        number of strings, etc.
    string_frequency_:
        dict[str, numbers.Integral] - The unique strings and the number
        of occurrences seen during fitting.
    starts_with_frequency_:
        dict[str, numbers.Integral] - A dictionary that holds the first
        characters and their frequencies in the first position for all
        the strings fitted on the TextStatistics instance.
    character_frequency_:
        dict[str, numbers.Integral] - A dictionary that holds all the
        unique single characters and their frequencies for all the
        strings fitted on the TextStatistics instance.


    Examples
    --------
    >>> from pybear.feature_extraction.text import TextStatistics
    >>> STRINGS = ['I am Sam', 'Sam I am', 'That Sam-I-am!', 'That Sam-I-am!',
    ...    'I do not like that Sam-I-am!']
    >>> TS = TextStatistics()
    >>> TS.fit(STRINGS)
    TextStatistics()
    >>> TS.size_
    5
    >>> TS.overall_statistics_['max_length']
    28
    >>> TS.overall_statistics_['average_length']
    14.4

    >>> STRINGS = ['a', 'a', 'b', 'c', 'c', 'c', 'd', 'd', 'e', 'f', 'f']
    >>> TS = TextStatistics()
    >>> TS.fit(STRINGS)
    TextStatistics()
    >>> TS.size_
    11
    >>> TS.string_frequency_
    {'a': 2, 'b': 1, 'c': 3, 'd': 2, 'e': 1, 'f': 2}
    >>> TS.uniques_
    ['a', 'b', 'c', 'd', 'e', 'f']
    >>> TS.overall_statistics_['max_length']
    1
    >>> TS.character_frequency_
    {'a': 2, 'b': 1, 'c': 3, 'd': 2, 'e': 1, 'f': 2}

    """


    _lp: numbers.Integral = 5
    _rp: numbers.Integral = 15


    def __init__(self) -> None:
        """Intialize a TextStatistics class."""
        pass


    # @properties v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v
    # size_ -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    @property
    def size_(self) -> numbers.Integral:

        """The number of strings fitted on the TextStatistics instance."""

        check_is_fitted(self)

        return self.overall_statistics_['size']


    @size_.setter
    def size_(self, value):
        raise AttributeError(f'size_ attribute is read-only')
    # END size_ -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

    # uniques_ -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    @property
    def uniques_(self) -> Sequence[str]:  # pizza want ndarray here?

        """
        A list of the unique strings fitted on the TextStatistics
        instance.

        """


        uniques = list(self.string_frequency_.keys())

        _val_uniques(uniques)

        return uniques


    @uniques_.setter
    def uniques_(self, value):
        raise AttributeError(f'overall_statistics_ attribute is read-only')
    # END uniques_ -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    # END @properties v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v


    def _reset(self):

        try:
            del self.string_frequency_
            del self.overall_statistics_
            del self.starts_with_frequency_
            del self.character_frequency_
        except:
            pass


    def get_params(self, deep: Optional[bool] = True):

        """
        A spoof get_params for ReprMixin functionality. TextStatistics
        does not have any init parameters.
        """

        return {}


    def partial_fit(
        self,
        STRINGS: Sequence[str],
        y: Optional[any] = None
    ) -> Self:

        """
        Batch-wise accumulation of statistics.


        Parameters
        ----------
        STRINGS:
            Sequence[str] - a single list-like vector of strings to
            report statistics for, cannot be empty. strings do not need
            to be in the Lexicon. Individual strings cannot have spaces
            and must be under 30 characters in length.
        y:
            Optional[any], default = None - a target for the data.
            Always ignored.


        Return
        ------
        -
            self


        """

        _val_strings(STRINGS)

        # string_frequency_ -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
        # this must be before overall_statistics
        _current_string_frequency: StringFrequencyType = \
            _build_string_frequency(
                STRINGS,
                case_sensitive=True
            )

        self.string_frequency_: StringFrequencyType = \
            _merge_string_frequency(
                _current_string_frequency,
                getattr(self, '_string_frequency', {})
            )

        del _current_string_frequency

        _val_string_frequency(self.string_frequency_)
        # END string_frequency_ -- -- -- -- -- -- -- -- -- -- -- -- -- --

        # overall_statistics_ -- -- -- -- -- -- -- -- -- -- -- -- -- --
        _current_overall_statistics: OverallStatisticsType = \
            _build_overall_statistics(
                STRINGS,
                case_sensitive=True
            )

        self.overall_statistics_: OverallStatisticsType = \
            _merge_overall_statistics(
                _current_overall_statistics,
                getattr(self, '_overall_statistics', {}),
                len(self.string_frequency_)
            )

        del _current_overall_statistics

        _val_overall_statistics(self.overall_statistics_)
        # END overall_statistics_ -- -- -- -- -- -- -- -- -- -- -- -- --

        # startswith_frequency -- -- -- -- -- -- -- -- -- -- -- -- -- --
        _current_starts_with_frequency: StartsWithFrequencyType = \
            _build_startswith_frequency(
                self.string_frequency_
            )
        self.starts_with_frequency_: StartsWithFrequencyType = \
            _merge_startswith_frequency(
                _current_starts_with_frequency,
                getattr(self, '_starts_with_frequency', {})
            )

        del _current_starts_with_frequency

        _val_startswith_frequency(self.starts_with_frequency_)
        # END startswith_frequency -- -- -- -- -- -- -- -- -- -- -- --


        # character_frequency -- -- -- -- -- -- -- -- -- -- -- -- -- --
        _current_character_frequency: CharacterFrequencyType = \
            _build_character_frequency(
                self.string_frequency_
            )

        self.character_frequency_: CharacterFrequencyType = \
            _merge_character_frequency(
                _current_character_frequency,
                getattr(self, '_character_frequency', {})
            )

        del _current_character_frequency

        _val_character_frequency(self.character_frequency_)
        # END character_frequency -- -- -- -- -- -- -- -- -- -- -- -- --

        return self


    def fit(
        self,
        STRINGS: Sequence[str],
        y: Optional[any] = None
    ) -> Self:

        """
        Get statistics for one sequence of strings.


        Parameters
        ----------
        STRINGS:
            Sequence[str] - a single list-like vector of strings to
            report statistics for, cannot be empty. Strings do not need
            to be in the Lexicon. Individual strings cannot have spaces
            and must be under 30 characters in length.
        y:
            Optional[any], default = None - a target for the data. Always
            ignored.


        Return
        ------
        -
            self


        """

        self._reset()

        return self.partial_fit(STRINGS)


    # OTHER METHODS v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^

    def print_overall_statistics(self) -> None:

        """Print the 'overall_statistics_' attribute to screen."""

        check_is_fitted(self)

        _print_overall_statistics(self.overall_statistics_, self._lp, self._rp)


    def print_starts_with_frequency(self) -> None:

        """Print the 'starts_with_frequency_' attribute to screen."""

        check_is_fitted(self)

        _print_starts_with_frequency(
            self.starts_with_frequency_, self._lp, self._rp
        )


    def print_character_frequency(self) -> None:

        """Print the 'character_frequency_' attribute to screen."""

        check_is_fitted(self)

        _print_character_frequency(self.character_frequency_, self._lp, self._rp)


    def print_string_frequency(
        self,
        n:Optional[numbers.Integral] = 10
    ) -> None:

        """
        Print the 'string_frequency_' attribute to screen.


        Parameters
        ----------
        n:
            Optional[numbers.Integral], default = 10 - the number of the
            most frequent strings to print to screen.


        Return
        ------
        -
            None

        """

        check_is_fitted(self)

        _print_string_frequency(self.string_frequency_, self._lp, self._rp, n)


    # longest_strings -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    def get_longest_strings(
        self,
        n: Optional[numbers.Integral] = 10
    ) -> LongestStringsType:

        """
        The longest strings seen by the TextStatistics instance during
        fitting.


        Parameters
        ----------
        n:
            Optional[numbers.Integral], default = 10 - the number of the
            top longest strings to return.


        Return
        ------
        -
            dict[str, numbers.Integral] - the top 'n' longest strings
            seen by the TextStatistics instance during fitting.


        """

        check_is_fitted(self)

        __ = _get_longest_strings(self.string_frequency_, n=n)

        # _val_string_frequency will work for this
        _val_string_frequency(__)

        return __


    def print_longest_strings(
        self,
        n: Optional[numbers.Integral] = 10
    ) -> None:

        """
        Print the longest strings in the 'string_frequency_' attribute
        to screen.


        Parameters
        ----------
        n:
            Optional[numbers.Integral], default = 10 - the number of top
            longest strings to print to screen.


        Return
        ------
        -
            None


        """

        check_is_fitted(self)

        _print_longest_strings(self.string_frequency_, self._lp, self._rp, n)
    # END longest_strings -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

    # shortest_strings -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    def get_shortest_strings(
        self,
        n: Optional[numbers.Integral] = 10
    ) -> ShortestStringsType:

        """
        The shortest strings seen by the TextStatistics instance during
        fitting.


        Parameters
        ----------
        n:
            Optional[numbers.Integral], default = 10 - the number of the
            top shortest strings to return.


        Return
        ------
        -
            dict[str, numbers.Integral] - the top 'n' shortest strings
            seen by the TextStatistics instance during fitting.


        """

        check_is_fitted(self)

        __ = _get_shortest_strings(self.string_frequency_, n=n)

        # _val_string_frequency will work for this
        _val_string_frequency(__)

        return __


    def print_shortest_strings(
        self,
        n: Optional[numbers.Integral] = 10
    ) -> None:

        """
        Print the shortest strings in the 'string_frequency_' attribute
        to screen.


        Parameters
        ----------
        n:
            Optional[numbers.Integral], default = 10 - the number of
            shortest strings to print to screen.

        Return
        ------
        -
            None

        """

        check_is_fitted(self)

        _print_shortest_strings(self.string_frequency_, self._lp, self._rp, n)
    # END shortest_strings -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

    def lookup_substring(
        self,
        char_seq: str,
        case_sensitive: Optional[bool] = True
    ) -> Sequence[str]:   # pizza want ndarray here?

        """
        Return a sequence of all strings that have been fitted on the
        TextStatistics instance that contain the given character
        substring.


        Parameters
        ----------
        char_seq:
            str - character substring to be looked up against the
            strings fitted on the TextStatistics instance.
        case_sensitive:
            Optional[bool], default = True - If True, search for the
            exact string in the fitted data. If False, normalize both
            the given string and the strings fitted on the TextStatistics
            instance, then perform the search.


        Return
        ------
        -
            matching_strings: Sequence[str] - sequence of all strings in
            the fitted data that contain the given character substring.
            Returns an empty sequence if there are no matches.


        """

        check_is_fitted(self)

        return _lookup_substring(char_seq, self.uniques_, case_sensitive)


    def lookup_string(
        self,
        char_seq: str,
        case_sensitive: Optional[bool]=False
    ) -> Sequence[str]:   # pizza want ndarray here?

        """
        Look in the fitted strings for a full character sequence (not a
        substring) that exactly matches the given character sequence. If
        the case_sensitive parameter is True, look for an identical match
        to the given character sequence, and if at least one is found,
        return that character string. If an exact match is not found,
        return None. If the case_sensitive parameter is False, normalize
        the strings seen by the TextStatistics instance and the given
        character string and search for matches. If matches are found,
        return a 1D sequence of the matches in their original form from
        the fitted data (there may be different capitalizations in the
        fitted data, so there may be multiple entries.) If no matches
        are found, return None.


        Parameters
        ----------
        char_seq:
            str - character string to be looked up against the strings
            fitted on the TextStatistics instance.
        case_sensitive:
            Optional[bool], default = True - If True, search for the
            exact string in the fitted data. If False, normalize both
            the given string and the strings fitted on the TextStatistics
            instance, then perform the search.


        Return
        ------
        -
            Union[str, Sequence[str], None] - if there are any matches,
            return the matching string(s) from the originally fitted
            data; if there are no matches, return None.


        """


        check_is_fitted(self)

        return _lookup_string(char_seq, self.uniques_, case_sensitive)


    def score(self, X: any, y: Optional[any] = None) -> None:

        """
        Dummy method to spoof dask Incremental and ParallelPostFit
        wrappers. Verified must be here for dask wrappers.
        """

        check_is_fitted(self)

        return






