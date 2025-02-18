# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import (
    Optional,
    Sequence,
)
from typing_extensions import Self, Union

import numbers

from ._validation._strings import _val_strings
from ._validation._store_uniques import _val_store_uniques
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
from ._print._startswith_frequency import _print_startswith_frequency
from ._print._string_frequency import _print_string_frequency
from ._print._character_frequency import _print_character_frequency
from ._print._longest_strings import _print_longest_strings
from ._print._shortest_strings import _print_shortest_strings

from ._get._get_longest_strings import _get_longest_strings
from ._get._get_shortest_strings import _get_shortest_strings

from ._lookup._lookup_substring import _lookup_substring
from ._lookup._lookup_string import _lookup_string

from ....base import (
    GetParamsMixin,
    ReprMixin,
    check_is_fitted
)



class TextStatistics(GetParamsMixin, ReprMixin):


    _lp: int = 5
    _rp: int = 15


    def __init__(
        self,
        store_uniques: Optional[bool] = True
    ) -> None:

        """
        Generate summary information about a 1D sequence, or multiple
        1D sequences, of strings. Statistics include:

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

        TextStatistics (TS) has 2 scikit-style methods, partial_fit and
        fit. It does not have a transform method because it does not
        mutate data, it only reports information about the strings in it.

        TS can be fit on a single batch of data via :method: fit, and
        can be fit in batches via :method: partial_fit. The fit method
        resets the instance with each call, that is, all information
        held within the instance prior is deleted and the new fit
        information repopulates. The partial_fit method, however, does
        not reset and accumulates information across all batches seen.
        This makes TS suitable for streaming data and batch-wise
        training, such as with a dask_ml Incremental wrapper.

        TS does have other methods that allow access to certain
        functionality, such as conveniently printing summary information
        from attributes to screen. See the methods section of the docs.

        TS accepts 1D list-likes containing only strings. This includes
        numpy arrays, python lists, sets, and tuples, and pandas series.

        TS is case-sensitive during fitting, always. This is a deliberate
        design choice so that users who want to differentiate between
        the same characters in different cases can do so. If you want
        your strings to be treated in a non-case-sensitive way, normalize
        the case of your strings prior to fitting on TS.

        The TS instance takes only one parameter, 'store_uniques'. More
        on that below. The pybear intends for the 'store_uniques'
        parameter to be set once at instantiation and not changed
        thereafter for that instance. This protects the integrity of the
        reported information. As such, TS does not have a 'set_params'
        method, but it does have a 'get_params' method. Advanced users
        may access and set the 'store_uniques' parameter directly on the
        instance, but the impacts of doing so in the midst of a series
        of partial fits or afterward is not tested. pybear does not
        recommend this technique; create a new instance with the desired
        setting and fit your data again.

        When the 'store_uniques' parameter is True, the TS instance
        retains a dictionary of all the unique strings it has seen
        during fitting. In this case, TS is able to yield all the
        information that it is designed to collect. This is ideal for
        situations with a 'small' number of unique strings, such as when
        fitting on tokens, where a recurrence of a unique will simply
        increment the count of that unique in the dictionary instead of
        creating a new entry.

        When the 'store_uniques' parameter is False, however, the unique
        strings seen during fitting are not stored. In this case, the
        memory footprint of the TS instance will not grow linearly with
        the number of unique strings seen during fitting. This enables
        TS to fit on practially unlimited amounts of text data. This is
        ideal for situations where the individual strings being fit are
        phrases, sentences, or even entire books. This comes at cost,
        though, because some reporting capability is lost.

        Functionality available when 'store_uniques' is False is size
        (the number of strings seen by the TS instance), average length,
        standard deviation of length, maximum length, minimum length,
        overall character frequency, and first character frequency.
        Functionality lost includes the unique strings themselves as
        would otherwise be available through the 'uniques_' and
        'string_frequencies_' attributes, longest string, shortest
        string, lookup substring, and lookup string reporting and
        printing.


        Parameters
        ----------
        store_uniques:
            Optional[bool], default = True - whether to retain the
            unique strings seen by the TextStatistics instance in memory.
            If True, all attributes and print methods are fully
            informative. If False, the 'string_frequencies_' and
            'uniques_' attributes are always empty, and functionality
            that depends on these attributes have reduced capability.


        Attributes
        ----------
        size_:
            int - The number of strings fitted on the TextStatistics
            instance.
        uniques_:
            list[str] - A 1D list of the unique strings fitted on the
            TextStatistics instance. If parameter 'store_uniques' is
            False, this will always be empty.
        overall_statistics_:
            dict[str: numbers.Real] - A dictionary that holds information
            about all the strings fitted on the TextStatistics instance.
            Available statistics are size (number of strings seen during
            fitting), uniques count, average string length, standard
            deviation of string length, maximum string length, and
            minimum string length. If parameter 'store_uniques' is False,
            the 'uniques_count' field will always be zero.
        string_frequency_:
            dict[str, int] - A dictionary that holds the unique strings
            and the respective number of occurrences seen during fitting.
            If parameter 'store_uniques' is False, this will always be
            empty.
        startswith_frequency_:
            dict[str, int] - A dictionary that holds the first characters
            and their frequencies in the first position for all the
            strings fitted on the TextStatistics instance.
        character_frequency_:
            dict[str, int] - A dictionary that holds all the unique
            single characters and their frequencies for all the strings
            fitted on the TextStatistics instance.


        Examples
        --------
        >>> from pybear.feature_extraction.text import TextStatistics
        >>> STRINGS = ['I am Sam', 'Sam I am', 'That Sam-I-am!',
        ...    'That Sam-I-am!', 'I do not like that Sam-I-am!']
        >>> TS = TextStatistics(store_uniques=True)
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


        self.store_uniques = store_uniques


    # @properties v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v
    @property
    def size_(self) -> int:

        """The number of strings fitted on the TextStatistics instance."""

        check_is_fitted(self)

        return self.overall_statistics_['size']


    @property
    def uniques_(self) -> list[str]:

        """
        list[str] - A 1D list of the unique strings fitted on the
        TextStatistics instance. If parameter 'store_uniques' is False,
        this will always be empty.
        """

        uniques = list(self.string_frequency_.keys())

        _val_uniques(uniques)

        return uniques
    # END @properties v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v


    def _reset(self):

        """
        Reset the TextStatistics instance to the not-fitted state,
        i.e., remove all objects that hold information from any fits that
        may have been performed on the instance.

        """

        try:
            del self.string_frequency_
            del self.overall_statistics_
            del self.startswith_frequency_
            del self.character_frequency_
        except:
            pass


    # def get_params() - inherited from GetParamsMixin


    def partial_fit(
        self,
        X: Sequence[str],
        y: Optional[any] = None
    ) -> Self:

        """
        Batch-wise fitting of the TextStatistics instance on string data.
        The instance is not reset and information about the strings in
        the batches of training data is accretive.


        Parameters
        ----------
        X:
            Sequence[str] - a single list-like vector of strings to
            report statistics for, cannot be empty. strings do not need
            to be in the pybear Lexicon. Individual strings cannot have
            spaces and must be under 30 characters in length.
        y:
            Optional[any], default = None - a target for the data.
            Always ignored.


        Return
        ------
        -
            self


        """

        _val_strings(X)
        _val_store_uniques(self.store_uniques)

        # string_frequency_ -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
        # this must be first
        _current_string_frequency: dict[str, numbers.Integral] = \
            _build_string_frequency(
                X,
                case_sensitive=True
            )

        if self.store_uniques:
            self.string_frequency_: dict[str, int] = \
                _merge_string_frequency(
                    _current_string_frequency,
                    getattr(self, 'string_frequency_', {})
                )
        else:
            self.string_frequency_ = {}

        _val_string_frequency(self.string_frequency_)
        # END string_frequency_ -- -- -- -- -- -- -- -- -- -- -- -- -- --

        # startswith_frequency -- -- -- -- -- -- -- -- -- -- -- -- -- --
        _current_startswith_frequency: dict[str, numbers.Integral] = \
            _build_startswith_frequency(
                _current_string_frequency
            )

        self.startswith_frequency_: dict[str, int] = \
            _merge_startswith_frequency(
                _current_startswith_frequency,
                getattr(self, 'startswith_frequency_', {})
            )

        del _current_startswith_frequency

        _val_startswith_frequency(self.startswith_frequency_)
        # END startswith_frequency -- -- -- -- -- -- -- -- -- -- -- --

        # character_frequency -- -- -- -- -- -- -- -- -- -- -- -- -- --
        _current_character_frequency: dict[str, numbers.Integral] = \
            _build_character_frequency(
                _current_string_frequency
            )

        self.character_frequency_: dict[str, int] = \
            _merge_character_frequency(
                _current_character_frequency,
                getattr(self, 'character_frequency_', {})
            )

        del _current_string_frequency
        del _current_character_frequency

        _val_character_frequency(self.character_frequency_)
        # END character_frequency -- -- -- -- -- -- -- -- -- -- -- -- --

        # overall_statistics_ -- -- -- -- -- -- -- -- -- -- -- -- -- --
        _current_overall_statistics = \
            _build_overall_statistics(
                X,
                case_sensitive=False
            )

        if not self.store_uniques:
            _current_overall_statistics['uniques_count'] = 0

        self.overall_statistics_: dict[str, numbers.Real] = \
            _merge_overall_statistics(
                _current_overall_statistics,
                getattr(self, 'overall_statistics_', {}),
                _len_uniques=len(self.uniques_)
            )

        _val_overall_statistics(self.overall_statistics_)
        # END overall_statistics_ -- -- -- -- -- -- -- -- -- -- -- -- --

        return self


    def fit(
        self,
        X: Sequence[str],
        y: Optional[any] = None
    ) -> Self:

        """
        Single batch training of the TextStatistics instance on string
        data. The instance is reset and the only information retained is
        that associated with this single batch of data.


        Parameters
        ----------
        X:
            Sequence[str] - a single list-like vector of strings to
            report statistics for, cannot be empty. Strings do not need
            to be in the pybear Lexicon.
        y:
            Optional[any], default = None - a target for the data. Always
            ignored.


        Return
        ------
        -
            self


        """

        self._reset()

        return self.partial_fit(X)


    def transform(self, X: Sequence[str]) -> Sequence[str]:

        """
        A no-op transform method for data processing scenarios that may
        require the transform method. X is returned as given.


        Parameters
        ----------
        X:
            Sequence[str] - the data. Ignored.


        Return
        ------
        -
            X: Sequence[str] - the original, unchanged, data.


        """

        return X


    # OTHER METHODS v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^

    def print_overall_statistics(self) -> None:

        """
        Print the 'overall_statistics_' attribute to screen. If parameter
        'store_uniques' is False, the 'uniques_count' field will always
        be zero.
        """

        check_is_fitted(self)

        _print_overall_statistics(self.overall_statistics_, self._lp, self._rp)


    def print_startswith_frequency(self) -> None:

        """Print the 'startswith_frequency_' attribute to screen."""

        check_is_fitted(self)

        _print_startswith_frequency(
            self.startswith_frequency_, self._lp, self._rp
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
        Print the 'string_frequency_' attribute to screen. Only available
        if parameter 'store_uniques' is True. If False, uniques are not
        available for display to screen.


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
    ) -> dict[str, int]:

        """
        The longest strings seen by the TextStatistics instance during
        fitting. Only available if parameter 'store_uniques' is True. If
        False, the uniques seen during fitting are not available and an
        empty dictionary is always returned.


        Parameters
        ----------
        n:
            Optional[numbers.Integral], default = 10 - the number of the
            top longest strings to return.


        Return
        ------
        -
            dict[str, int] - the top 'n' longest strings seen by the
            TextStatistics instance during fitting. Empty if the
            'store_uniques' parameter is False.


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
        to screen. Only available if parameter 'store_uniques' is True.
        If False, uniques are not available for display to screen.


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
    ) -> dict[str, int]:

        """
        The shortest strings seen by the TextStatistics instance during
        fitting. Only available if parameter 'store_uniques' is True. If
        False, the uniques seen during fitting are not available and an
        empty dictionary is always returned.


        Parameters
        ----------
        n:
            Optional[numbers.Integral], default = 10 - the number of the
            top shortest strings to return.


        Return
        ------
        -
            dict[str, int] - the top 'n' shortest strings seen by the
            TextStatistics instance during fitting. Empty if the
            'store_uniques' parameter is False.


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
        to screen. Only available if parameter 'store_uniques' is True.
        If False, uniques are not available for display to screen.


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
    ) -> list[str]:

        """
        Return a list of all strings that have been fitted on the
        TextStatistics instance that contain the given character
        substring. This is only available if parameter 'store_uniques'
        is True. If False, the unique strings that have been fitted on
        the TextStatistics instance are not retained therefore cannot be
        searched, so any empty list is returned.


        Parameters
        ----------
        char_seq:
            str - character substring to be looked up against the strings
            fitted on the TextStatistics instance.
        case_sensitive:
            Optional[bool], default = True - If True, search for the
            exact string in the fitted data. If False, normalize both
            the given string and the strings fitted on the TextStatistics
            instance, then perform the search.


        Return
        ------
        -
            list[str] - list of all strings in the fitted data that
            contain the given character substring. Returns an empty list
            if there are no matches.


        """

        check_is_fitted(self)

        return _lookup_substring(char_seq, self.uniques_, case_sensitive)


    def lookup_string(
        self,
        char_seq: str,
        case_sensitive: Optional[bool]=True
    ) -> Union[str, list[str], None]:

        """
        Look in the fitted strings for a full character sequence (not a
        substring) that exactly matches the given character sequence. If
        the case_sensitive parameter is True, look for an identical match
        to the given character sequence, and if at least one is found,
        return that character string. If an exact match is not found,
        return None. If the case_sensitive parameter is False, normalize
        the strings seen by the TextStatistics instance and the given
        character string and search for matches. If matches are found,
        return a 1D list of the matches in their original form from the
        fitted data (there may be different capitalizations in the
        fitted data, so there may be multiple entries.) If no matches
        are found, return None.

        This is only available if parameter 'store_uniques' is True. If
        False, the unique strings that have been fitted on the
        TextStatistics instance are not retained therefore cannot be
        searched, and None is always returned.


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
            Union[str, list[str], None] - if there are any matches,
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









