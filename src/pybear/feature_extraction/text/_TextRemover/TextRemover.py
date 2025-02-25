# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Optional
from typing_extensions import Self, Union
from ._type_aliases import (
    XContainer,
    StrRemoveType,
    RegExpRemoveType,
    RegExpFlagsType
)

from copy import deepcopy

from ....base import (
    FitTransformMixin,
    GetParamsMixin,
    ReprMixin,
    SetParamsMixin,
    check_is_fitted
)

from ._transform._transform import _transform
from ._validation._validation import _validation



class TextRemover(
    FitTransformMixin,
    GetParamsMixin,
    ReprMixin,
    SetParamsMixin
):


    def __init__(
        self,
        *,
        str_remove: Optional[StrRemoveType] = None,
        regexp_remove: Optional[RegExpRemoveType] = None,
        regexp_flags: Optional[RegExpFlagsType] = None
    ) -> None:

        """
        Remove full strings (not substrings) from text data.

        Identify full strings to remove by ordinary string comparison
        (== or list.remove) or by regular expression match (re.fullmatch).
        Remove any and all matches completely from the data. For 2D
        array-like data, empty rows will be removed also, whether they
        were given as empty, or became empty by the string removal
        process. It is possible that empty 1D lists are returned.
        Comparisons using 'str_remove' are always case-sensitive.

        One particularly useful application is to take out empty lines
        and such that may have been in data read in from a file, or
        strings that become empty after replacing values.

        TextRemover is a full-fledged scikit-style transformer.
        get_params, set_params, no-op partial_fit, no-op fit, transform,
        fit_transform, no-op score.

        Accepts 1D list-like and (possibly ragged) 2D array-likes of
        strings. Does not accept pandas dataframes. Convert your data to
        a python list of lists or a numpy array. When passed a 1D
        list-like, returns a python list of strings. When passed a 2D
        array-like, returns a python list of python lists of strings.


        TypeAliases
        -----------
        XContainer:
            TypeAlias = Union[Sequence[str], Sequence[Sequence[str]]]

        StrType:
            TypeAlias = Union[str, set[str]]
        StrRemoveType:
            TypeAlias = Union[None, StrType, list[Union[StrType, Literal[False]]]]

        RegExpType:
            TypeAlias = Union[str, re.Pattern]
        RegExpRemoveType: TypeAlias = \
            Union[None, RegExpType, list[Union[RegExpType, Literal[False]]]]

        FlagType: TypeAlias = \
            Union[None, numbers.Integral]
        RegExpFlagsType: TypeAlias = \
            Union[FlagType, list[Union[FlagType, Literal[False]]]]


        Parameters
        ----------
        str_remove:
            Optional[StrSepType], default=None - the patterns to remove
            from X when in exact string matching mode. When passed as a
            single character string, that is applied to every string in
            X, and every full string that matches it exactly will be
            removed. When passed as a set of character strings, each
            string is searched against all the strings in X. If passed
            as a list of strings, the number of entries must match the
            number of strings in X, and each string or set of strings is
            applied to the corresponding string in X. If any entry in
            the list is False, the corresponding string in X is skipped.
        regexp_remove:
            Optional[RegExpSepType], default=None - if using regular
            expressions, the regexp pattern(s) to remove from X. If a
            single regular expression or re.Patten object is passed,
            any matching full strings in X will be removed from the data.
            If passed as a list, the number of entries must match the
            number of strings in X, and each pattern is applied to the
            corresponding string in X. If any entry in the list is False,
            that string in X is skipped.
        regexp_flags:
            Optional[RegExpFlagsType] - the flags parameter for re.split,
            if regular expressions are being used. Only applies if a
            pattern is passed to 'regexp_remove'. If None, the default
            flags for re.split() are used on every string in X. If a
            single flags object, that is applied to every string in X.
            If passed as a list, the number of entries must match the
            number of strings in X. Flags objects and Nones in the list
            follow the same rules stated above. If any entry in the list
            is False, that string in X is skipped.

        """

        self.str_remove = str_remove
        self.regexp_remove = regexp_remove
        self.regexp_flags = regexp_flags


    def __pybear_is_fitted__(self):
        return True


    # def get_params
    # from GetParamsMixin


    # def set_params
    # from SetParamsMixin


    # def fit_transform
    # from FitTransformMixin


    def partial_fit(
        self,
        X: XContainer,
        y: Optional[Union[any, None]] = None
    ) -> Self:

        """
        Batch-wise no-op fit operation.


        Parameters
        ----------
        X:
            list-like 1D vector of strings or (possibly ragged) 2D
            array-like of strings - the data.
        y:
            Optional[Union[any, None]], default=None - the target for
            the data. Always ignored.


        Return
        ------
        -
            self - the TextRemover instance.


        """

        check_is_fitted(self)

        return self


    def fit(
        self,
        X: XContainer,
        y: Optional[Union[any, None]] = None
    ) -> Self:

        """
        One-shot no-op fit operation.


        Parameters
        ----------
        X:
            list-like 1D vector of strings or (possibly ragged) 2D
            array-like of strings - the data.
        y:
            Optional[Union[any, None]], default=None - the target for
            the data. Always ignored.


        Return
        ------
        -
            self - the TextRemover instance.

        """

        check_is_fitted(self)

        return self.partial_fit(X, y)


    def transform(
        self,
        X: XContainer,
        copy: Optional[bool] = True
    ) -> XContainer:

        """
        Remove unwanted strings from the data.


        Parameters
        ----------
        X:
            list-like 1D vector of strings or (possibly ragged) 2D
            array-like of strings - the data.
        copy:
            Optional[bool], default=True - whether to make a copy of the
            data before removing the unwanted strings.


        Return
        ------
        _X:
            Union[list[str], list[list[str]] - the data with unwanted
            strings removed.


        """

        check_is_fitted(self)

        _validation(
            X,
            self.str_remove,
            self.regexp_remove,
            self.regexp_flags
        )

        if copy:
            if isinstance(X, (list, tuple, set)) or not hasattr(X, 'copy'):
                _X = deepcopy(X)
            else:
                _X = X.copy()
        else:
            _X = X

        if isinstance(_X[0], str):
            _X = list(_X)
        else:
            # must be 2D
            _X = list(map(list, _X))


        return _transform(
            _X,
            self.str_remove,
            self.regexp_remove,
            self.regexp_flags
        )


    def score(
        self,
        X: XContainer,
        y: Optional[Union[any, None]] = None
    ) -> None:

        """
        No-op score method to allow wrap by dask_ml wrappers.


        Parameters
        ----------
        X:
            list-like 1D vector of strings or (possibly ragged) 2D
            array-like of strings - the data. Always ignored.
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



