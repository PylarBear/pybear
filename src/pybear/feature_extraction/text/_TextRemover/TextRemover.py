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
    RegExpFlagsType,
    RowSupportType
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

    """
    Remove full strings (not substrings) from text data.

    One particularly useful application is to take out empty lines and
    strings that may have been in data read in from a file, or strings
    that have become empty or have only non-alphanumeric characters after
    replacing values (see pybear TextReplacer).

    Identify full strings to remove by ordinary string comparison
    (== or list.remove) or by regular expression match (re.fullmatch).
    Remove any and all matches completely from the data. For 2D
    array-like data, empty rows will be removed also, whether they were
    given as empty, or became empty by the string removal process. It is
    possible that empty 1D lists are returned. Comparisons using
    'str_remove' are always case-sensitive.

    Direct string comparison mode and regular expression match mode are
    mutually exclusive, they cannot both be used at the same time.
    Parameters for only one or the other can be passed at instantiation,
    i.e., only 'str_remove' can be entered, or only 'regexp_remove' and
    'regexp_flags" can be entered. At least one mode must be indicated
    at instantiation, TextRemover cannot be instantiated with the default
    parameter values.

    TextRemover is a full-fledged scikit-style transformer. It has fully
    functional get_params, set_params, transform, and fit_transform
    methods. It also has no-op partial_fit and fit methods to allow for
    integration into larger workflows. Technically TextRemover does not
    need to be fit and is always in a fitted state (any 'is_fitted'
    checks of an instance will always return True) because TextRemover
    knows everything it needs to know to do transformations from the
    parameters. It also has a no-op score method to allow dask_ml
    wrappers.

    Accepts 1D list-like and (possibly ragged) 2D array-likes of strings.
    Does not accept pandas dataframes. Convert your pandas dataframe to
    a python list of lists or a numpy array. When passed a 1D list-like,
    returns a python list of strings. When passed a 2D array-like,
    returns a python list of python lists of strings.

    TextRemover instances that have undergone a transform operation have
    a 'row_support_' attribute. This is a boolean numpy array indicating
    which rows were kept in the data during the last transform. True
    indicates the row was kept and False indicates the row was removed.
    This mask can be applied to a target for the data (if any) so that
    the rows in the target match the rows in the data after transform.


    Parameters
    ----------
    str_remove:
        Optional[StRemoveSepType], default=None - the strings to remove
        from X when in exact string matching mode. Always case-sensitive.
        When passed as a single character string, that is applied to
        every string in X, and every full string that matches it exactly
        will be removed. When passed as a python set of character
        strings, each string is searched against all the strings in X,
        and any exact matches are removed. If passed as a list of
        strings, the number of entries must match the number of rows in
        X, and each string or set of strings in the list is applied to
        the corresponding string in X. If any entry in the list is False,
        the corresponding string in X is skipped.
    regexp_remove:
        Optional[RegExpRemoveType], default=None - if using regular
        expressions, the regexp pattern(s) to remove from X. If a single
        regular expression or re.Patten object is passed, any matching
        full strings in X will be removed from the data. If passed as a
        list, the number of entries must match the number of rows in X,
        and each pattern is applied to the corresponding string in X. If
        any entry in the list is False, that string in X is skipped.
    regexp_flags:
        Optional[RegExpFlagsType] - the flags parameter(s) for the regexp
        pattern(s), if regular expressions are being used. Does not apply
        to string mode, only applies if a pattern is passed to
        'regexp_remove'. If None, the default flags for re.fullmatch()
        are used on every string in X. If a single flags object, that is
        applied to every string in X. If passed as a list, the number of
        entries must match the number of rows in X. Flags objects and
        Nones in the list follow the same rules stated above. If any
        entry in the list is False, that string in X is skipped.


    Attributes
    ----------
    row_support_:
        RowSupportType - A boolean vector indicating which rows were kept
        in the data during the transform process. Only available if a
        transform has been performed, and only reflects the results of
        the last transform done.


    Notes
    -----
    Type Aliases

    XContainer:
        Union[Sequence[str], Sequence[Sequence[str]]]

    StrType:
        Union[str, set[str]]
    StrRemoveType:
        Union[None, StrType, list[Union[StrType, Literal[False]]]]

    RegExpType:
        Union[str, re.Pattern]
    RegExpRemoveType:
        Union[None, RegExpType, list[Union[RegExpType, Literal[False]]]]

    FlagType:
        Union[None, numbers.Integral]
    RegExpFlagsType:
        Union[FlagType, list[Union[FlagType, Literal[False]]]]

    RowSupportType:
        numpy.typing.NDArray[bool]


    See Also
    --------
    list.remove
    re.fullmatch


    Examples
    --------
    >>> from pybear.feature_extraction.text import TextRemover as TR
    >>> trfm = TR(str_remove={' ', ''})
    >>> X = [' ', 'One', 'Two', '', 'Three', ' ']
    >>> trfm.fit_transform(X)
    ['One', 'Two', 'Three']
    >>> trfm.set_params(**{'str_remove': None, 'regexp_remove': '[bcdei]'})
    TextRemover(regexp_remove='[bcdei]')
    >>> X = [['a', 'b', 'c'], ['d', 'e', 'f'], ['g', 'h', 'i']]
    >>> trfm.fit_transform(X)
    [['a'], ['f'], ['g', 'h']]


    """


    def __init__(
        self,
        *,
        str_remove: Optional[StrRemoveType] = None,
        regexp_remove: Optional[RegExpRemoveType] = None,
        regexp_flags: Optional[RegExpFlagsType] = None
    ) -> None:

        self.str_remove = str_remove
        self.regexp_remove = regexp_remove
        self.regexp_flags = regexp_flags


    def __pybear_is_fitted__(self):
        return True


    @property
    def row_support_(self) -> RowSupportType:
        """
        A boolean vector indicating which rows were kept in the data
        during the transform process. Only available if a transform has
        been performed, and only reflects the results of the last
        transform done.
        """

        return self._row_support


    def get_metadata_routing(self):
        raise NotImplementedError(
            f"metadata routing is not implemented in TextRemover"
        )


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
            Optional[bool], default=True - whether to remove the unwanted
            strings directly from the original X or from a copy of the
            original X.


        Return
        ------
        -
            Union[list[str], list[list[str]]] - the data with unwanted
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

        if all(map(isinstance, _X, (str for _ in _X))):
            _X = list(_X)
        else:
            # must be 2D
            _X = list(map(list, _X))


        _X, self._row_support = _transform(
            _X,
            self.str_remove,
            self.regexp_remove,
            self.regexp_flags
        )


        return _X


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



