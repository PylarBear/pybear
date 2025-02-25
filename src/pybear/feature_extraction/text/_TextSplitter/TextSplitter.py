# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Optional
from typing_extensions import Self, Union
from ._type_aliases import (
    XContainer,
    StrSepType,
    RegExpSepType,
    StrMaxSplitType,
    RegExpMaxSplitType,
    RegExpFlagsType
)

from copy import deepcopy

from ....base import (
    FitTransformMixin,
    GetParamsMixin,
    ReprMixin,
    SetParamsMixin
)

from ._validation._validation import _validation
from ._transform._str_core import _str_core
from ._transform._regexp_core import _regexp_core



class TextSplitter(
    FitTransformMixin,
    GetParamsMixin,
    ReprMixin,
    SetParamsMixin
):


    def __init__(
        self,
        *,
        str_sep: Optional[StrSepType] = None,
        str_maxsplit: Optional[StrMaxSplitType] = None,
        regexp_sep: Optional[RegExpSepType] = None,
        regexp_maxsplit: Optional[RegExpMaxSplitType] = None,
        regexp_flags: Optional[RegExpFlagsType] = None
    ):

        """
        Split a dataset of strings on the given separator(s).

        TextSplitter has 2 independent splitting modes that use Python
        built-in functions, one uses str.split() and the other uses
        re.split().

        The 2 modes cannot be used simultaneously. Enter values for only
        those parameters prefixed by 'str_' to use str.split(), or enter
        values for only those parameters prefixed by 'regexp_' to use
        re.split(). If no parameters are passed, i.e., all parameters
        are left to their default values of None, then TextSplitter uses
        the default splitting for str.split() on every string in the
        data.

        If in str.split() mode and 'str_maxsplit' is None, the default
        number of splits for str.split() are performed. If in re.split()
        mode and 'regexp_maxsplit' or 'regexp_flags' is None, then the
        default number of splits and the default flags for re.split()
        are used.

        So why not just use str.split or re.split()? TextSplitter has
        some advantages over the built-ins.

        First, in str.split() mode, multiple splitting criteria can be
        passed to the 'str_sep' parameter to split on multiple character
        sequences, which str.split() cannot do natively. For example,
        consider the string "How, now. brown; cow?". This can be split
        on the comma, period, and semicolon by passing a set to the
        'str_sep' parameter, such as {',', '.', ';'}. The output will be
        ["How", " now", " brown", " cow?"].

        Second, the splitting criteria for both splitting modes are
        simultaneously mapped over a list of strings, performing  many
        splits in a single operation. Both str.split() and re.split()
        only accept one string argument.

        Third, the split criteria and supporting parameters can be
        tweaked for individual strings in the data by passing them as
        lists. This allows fine-grained control over splitting every
        string in the data.

        TextSplitter is a full-fledged scikit-style transformer. The only
        operative method is 'transform', which accepts 1D list-likes of
        strings. It has no-op 'partial_fit', 'fit', and 'score' methods,
        so that it integrates into larger workflows like scikit pipelines
        and dask_ml wrappers. It also has 'get_params' and 'set_params'
        methods.

        When performing multiple split criteria in str.split() mode,
        i.e., you have passed a set of string characters to the 'str_sep'
        parameter, the 'str_maxsplit' parameter is applied cumulatively
        for all separators working from left to right across the strings
        in the data. For example, consider the string
        "One, two, buckle my shoe. Three, four, shut the door.". We are
        going to split on commas and periods, and perform 4 splits,
        working from left to right. We enter parameter 'str_sep' as
        {',', '.'} and parameter 'str_maxsplits' as 4. Then we pass the
        string in a list to the 'transform' method of TextSplitter. The
        output will be
        ["One", " two", " buckle my shoe", " Three", " four, shut the door."]
        The 'str_maxsplit' argument worked from left to right and
        performed 4 splits on commas and periods cumulatively counting
        the application of the splits for all separators.


        Type Aliases
        ------------
        XContainer: Sequence[str]

        SepType: Union[str, set[str], None]

        StrSepType: Union[SepType, list[Union[SepType, Literal[False]]]]

        RegExpType: Union[str, re.Pattern]

        RegExpSepType:
            Union[RegExpType, None, list[Union[RegExpType, Literal[False]]]]

        MaxSplitType: Union[numbers.Integral, None]

        StrMaxSplitType:
            Union[MaxSplitType, list[Union[MaxSplitType, Literal[False]]]]

        RegExpMaxSplitType:
            Union[MaxSplitType, list[Union[MaxSplitType, Literal[False]]]]

        FlagType: Union[numbers.Integral, None]

        RegExpFlagsType: \
            Union[FlagType, list[Union[FlagType, Literal[False]]]]


        Parameters
        ----------
        str_sep:
            Optional[StrSepType], default=None - the separator(s) to
            split the strings in X on when in str.split() mode. None
            applies the default str.split() criteria to every string in
            X. When passed as a single character string, that is applied
            to every string in X. When passed as a set of character
            strings, each separator in the set is applied to every
            string. If passed as a list of separators, the number of
            entries must match the number of strings in X, and each
            string or set of strings is applied to the corresponding
            string in X. If any entry in the list is False, no split is
            performed on the corresponding string in X. Case-sensitive.
        str_maxsplit:
            Optional[StrMaxSplitType], default=None - the maximum number
            of splits to perform when in str.split() mode. Only applies
            when something is passed to 'str_sep'. If None, the default
            number of splits for str.split() is used on every string in
            X. If passed as an integer, that number is applied to every
            string in X. If passed as a list, the number of entries must
            match the number of strings in X, and each is applied
            correspondingly to X, subject to the rules for Nones and
            numbers stated above.  If any entry in the list is False, no
            split is performed on the corresponding string in X.
        regexp_sep:
            Optional[RegExpSepType], default=None - if using regular
            expressions, the regexp pattern(s) to split the strings in X
            on. If a single regular expression or re.Patten object is
            passed, that split is performed on every entry in X. If
            passed as a list, the number of entries must match the number
            of strings in X, and each pattern is applied to the
            corresponding string in X. If any entry in the list is False,
            no split is performed for that string in X.
        regexp_maxsplit:
            Optional[RegExpMaxSplitType], default=None - the maximum
            number of splits to perform. Only applies if a pattern is
            passed to 'regexp_sep'. If None, the default number of splits
            for re.split() are performed. If passed as a list, the number
            of entries must match the number of strings in X. Integers
            and Nones in the list follow the same rules stated above. If
            any entry in the list is False, no split is performed for
            that string in X.
        regexp_flags:
            Optional[RegExpFlagsType] - the flags parameter for re.split,
            if regular expressions are being used. Only applies if a
            pattern is passed to 'regexp_sep'. If None, the default flags
            for re.split() are used on every string in X. If a single
            flags object, that is applied to every string in X. If passed
            as a list, the number of entries must match the number of
            strings in X. Flags objects and Nones in the list follow the
            same rules stated above. If any entry in the list is False,
            no split is performed for that string in X.


        Notes
        -----
        str.split()
        re.split()


        Examples
        --------
        >>> from pybear.feature_extraction.text import TextSplitter as TS
        >>> Trfm = TextSplitter(str_sep=' ', str_maxsplit=2)
        >>> X = [
        ...     'This is a test.',
        ...     'This is only a test.'
        ... ]
        >>> Trfm.fit(X)
        TextSplitter(str_maxsplit=2, str_sep=' ')
        >>> Trfm.transform(X)
        [['This', 'is', 'a test.'], ['This', 'is', 'only a test.']]

        >>> Trfm = TextSplitter(regexp_sep='s', regexp_maxsplit=2)
        >>> X = [
        ...     'This is a test.',
        ...     'This is only a test.'
        ... ]
        >>> Trfm.fit(X)
        TextSplitter(regexp_maxsplit=2, regexp_sep='s')
        >>> Trfm.transform(X)
        [['Thi', ' i', ' a test.'], ['Thi', ' i', ' only a test.']]



        """

        self.str_sep = str_sep
        self.str_maxsplit = str_maxsplit
        self.regexp_sep = regexp_sep
        self.regexp_maxsplit = regexp_maxsplit
        self.regexp_flags = regexp_flags


    # handled by mixins
    # def set_params
    # def get_params
    # def fit_transform


    def __pybear_is_fitted__(self):
        return True


    def get_metadata_routing(self):
        raise NotImplementedError(
            f"metadata routing is not implemented in TextSplitter"
        )


    def partial_fit(
        self,
        X: XContainer,
        y: Optional[Union[any, None]] = None
    ) -> Self:

        """
        No-op batch-wise fitting of TextSplitter.


        Parameters
        ----------
        X:
            list-like of shape (n_samples, ) - a 1D sequence of strings
            to be split.
        y:
            Optional[Union[any, None]], default=None - the target for
            the data.


        Return
        ------
        -
            self - the TextSplitter instance.


        """


        return self


    def fit(
        self,
        X: XContainer,
        y: Optional[Union[any, None]] = None
    ) -> Self:

        """
        No-op one-shot fitting of TextSplitter.


        Parameters
        ----------
        X:
            list-like of shape (n_samples, ) - a 1D sequence of strings
            to be split.
        y:
            Optional[Union[any, None]], default=None - the target for
            the data.


        Return
        ------
        -
            self - the TextSplitter instance.


        """


        return self.partial_fit(X, y)


    def transform(
        self,
        X: XContainer,
        copy: Optional[bool]=True
    ) -> list[list[str]]:

        """
        Split the strings in X on the separator(s).


        Parameters
        ----------
        X:
            list-like of shape (n_samples, ) - a 1D sequence of strings
            to be split.
        copy:
            Optional[bool] - whether to make a copy of X before performing
            the splits.


        Return
        ------
        -
            _X: list[list[str]] - the split strings.


        """

        _validation(
            X,
            self.str_sep,
            self.str_maxsplit,
            self.regexp_sep,
            self.regexp_maxsplit,
            self.regexp_flags
        )


        if copy:
            if isinstance(X, (list, tuple, set)) or not hasattr(X, 'copy'):
                _X = list(deepcopy(X))
            else:
                _X = list(X.copy())
        else:
            _X = list(X)

        _str_mode = False

        _a = bool(self.str_sep)
        _b = bool(self.str_maxsplit)
        _c = bool(self.regexp_sep)
        _d = bool(self.regexp_maxsplit)
        _e = bool(self.regexp_flags)

        if any((_a, _b)) or not any((_a, _b, _c, _d, _e)):
            _str_mode = True

        if _str_mode:

            _X = _str_core(
                _X,
                self.str_sep,
                self.str_maxsplit
            )

        elif not _str_mode:  # regexp

            _X = _regexp_core(
                _X,
                self.regexp_sep,
                self.regexp_maxsplit,
                self.regexp_flags
            )


        return _X


    def score(
        self,
        X: XContainer,
        y: Optional[Union[any, None]] = None
    ) -> None:

        """
        No-op scorer.


        Parameters
        ----------
        X:
            list-like of shape (n_samples, ) - a 1D sequence of strings.
        y:
            Optional[Union[any, None]], default=None - the target for
            the data.


        Return
        ------
        -
            None


        """


        return None








