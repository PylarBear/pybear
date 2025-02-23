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
        Split all the strings on the given separator(s). TextSplitter
        has 2 independent splitting modes that use Python built-in
        functions, one uses str.split() and the other uses re.split().


        # pizza talk about maxsplit and sets.


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

        RegExpType: Union[numbers.Integral, None]

        RegExpFlagsType: \
            Union[RegExpType, list[Union[RegExpType, Literal[False]]]]


        Parameters
        ----------
        str_sep:
            Optional[StrSepType], default=None - the separator(s) to
            split the strings in X on when in str.split() mode. When
            passed as a single character string or a set of such, that
            is applied to every string in X. None applies the default
            str.split() criteria to every string. If passed as a list of
            separators, the number of entries must match the number of
            strings in X.
        str_maxsplit:
            Optional[StrMaxSplitType], default=None - the maximum number
            of splits to perform when in str.split() mode. If passed as
            a list of integers, the number of entries must match the
            number of strings in X.
        regexp_sep:
            Optional[RegExpSepType], default=None - if using
            regular expressions, the regexp pattern(s) to split the
            strings in X on.
        regexp_maxsplit:
            Optional[RegExpMaxSplitType], default=None - the maximum
            number of splits to perform. If
            passed as a sequence of integers, the number of entries must
            match the number of strings in X.
        regexp_flags:
            Optional[RegExpFlagsType] - the flags parameter for re.split,
            if regular expressions are being used.


        Notes
        -----
        str.split()
        re.split()


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
            if isinstance(X, (list, set, tuple)):
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








