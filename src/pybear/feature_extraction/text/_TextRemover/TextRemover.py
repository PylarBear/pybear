# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Optional
from typing_extensions import Self, Union
from ._type_aliases import (
    XContainer,
    XWipContainer,
    StrRemoveType,
    RegExpRemoveType,
    RegExpFlagsType,
    RowSupportType
)

import pandas as pd
import polars as pl

from ._transform._transform import _transform
from ._validation._validation import _validation

from ....base import (
    FitTransformMixin,
    GetParamsMixin,
    ReprMixin,
    SetParamsMixin,
    check_is_fitted
)

from ....base._copy_X import copy_X



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

    Identify full strings to remove by literal string comparison
    (== or list.remove) or by regular expression match (re.fullmatch).
    Remove any and all matches completely from the data. For 2D
    array-like data, empty rows will be removed also, whether they were
    given as empty, or became empty by the string removal process. It is
    possible that empty 1D lists are returned. Comparisons using
    'str_remove' are always case-sensitive.

    Direct string comparison mode and regular expression match mode are
    mutually exclusive, they cannot both be used at the same time.
    Parameters for only one or the other can be passed at instantiation,
    For exact literal string matching, only :param: `str_remove` can
    be entered. To use regex, pass values to only :param: `regexp_remove`
    and :param: `regexp_flags`. At least one mode must be indicated at
    instantiation, TextRemover cannot be instantiated with the default
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
    Accepted 1D containers include python lists, tuples, and sets, numpy
    vectors, pandas series, and polars series. Accepted 2D containers
    include embedded python sequences, numpy arrays, pandas dataframes,
    and polars dataframes.  When passed a 1D list-like, returns a python
    list of strings. When passed a 2D array-like, returns a python list
    of python lists of strings. If you pass your data as a dataframe
    with feature names, the feature names are not preserved.

    TextRemover instances that have undergone a :term: transform
    operation have a :attr: `row_support_` attribute. This is a boolean
    numpy vector indicating which rows were kept (True) and which were
    removed (False) fram the data during the last :term: transform.
    This mask can be applied to a target for the data (if any) so that
    the rows in the target match the rows in the data after transform.
    The :attr: `row_support_` attribute only reflects the last dataset
    passed to transform.


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
        expressions, the regex pattern(s) to remove from X. If a single
        regular expression or re.Patten object is passed, any matching
        full strings in X will be removed from the data. If passed as a
        list, the number of entries must match the number of rows in X,
        and each pattern is applied to the corresponding string in X. If
        any entry in the list is False, that string in X is skipped.
    regexp_flags:
        Optional[RegExpFlagsType] - the flags parameter(s) for the regex
        pattern(s), if regular expressions are being used. Does not
        apply to string mode, only applies if a pattern is passed to
        the :param: `regexp_remove` parameter. If None, the default flags
        for re.fullmatch() are used on every string in X. If a single
        flags object, that is applied to every string in X. If passed as
        a list, the number of entries must match the number of rows in X.
        Flags objects and Nones in the list follow the same rules stated
        above. If any entry in the list is False, that string in X is
        skipped.


    Attributes
    ----------
    n_rows_:
        int - the number of rows in the data passed to :meth: `transform`.
        This reflects the data that is passed, not the data that is
        returned, which may not necessarily have the same number of
        rows as the original data. Also, it only reflects the last batch
        of data passed; it is not cumulative. This attribute is only
        exposed after data is passed to :meth: `transform`.
    row_support_:
        RowSupportType - A boolean vector indicating which rows were kept
        in the data during the :term: transform process. Only available
        if a transform has been performed, and only reflects the results
        of the last transform done.


    Notes
    -----
    Type Aliases

    PythonTypes:
        Union[Sequence[str], Sequence[Sequence[str]]]

    NumpyTypes:
        npt.NDArray[str]

    PandasTypes:
        Union[pd.Series, pd.DataFrame]

    PolarsTypes:
        Union[pl.Series, pl.DataFrame]

    XContainer:
        Union[PythonTypes, NumpyTypes, PandasTypes, PolarsTypes]

    XWipContainer:
        Union[list[str], list[list[str]]]

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
    def n_rows_(self):
        """
        Get the 'n_rows_' attribute. The number of rows in the data
        passed to transform.
        """
        return self._n_rows


    @property
    def row_support_(self):
        """
        Get the row_support_ attribute. A boolean vector indicating
        which rows were kept in the data during the transform process.
        Only available if a transform has been performed, and only
        reflects the results of the last transform done.
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


        return self.partial_fit(X, y)


    def transform(
        self,
        X: XContainer,
        copy: Optional[bool] = True
    ) -> XWipContainer:

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
            _X = copy_X(X)
        else:
            _X = X

        if all(map(isinstance, _X, (str for _ in _X))):
            _X = list(_X)
        else:
            # must be 2D
            if isinstance(_X, pd.DataFrame):
                _X = list(map(list, _X.values))
            elif isinstance(_X, pl.DataFrame):
                _X = list(map(list, _X.rows()))
            else:
                _X = list(map(list, _X))

        self._n_rows = len(_X)

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



