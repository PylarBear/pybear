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

import numpy as np
import pandas as pd
import polars as pl

from ._transform._str_1D_core import _str_1D_core
from ._transform._regexp_1D_core import _regexp_1D_core
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
    Remove any and all matches completely from the data.

    Direct string comparison mode and regular expression match mode
    are mutually exclusive, they cannot both be used at the same time.
    Parameters for only one or the other can be passed at instantiation.
    TextRemover cannot be instantiated with the default parameter values,
    at least one mode must be indicated. For exact literal string
    matching, only enter values for :param: `str_remove`. Comparisons
    in literal string mode are always case-sensitive. To use regex, pass
    values to only :param: `regexp_remove` and :param: `regexp_flags`.

    TextRemover is a full-fledged scikit-style transformer. It has fully
    functional get_params, set_params, transform, and fit_transform
    methods. It also has no-op partial_fit and fit methods to allow for
    integration into larger workflows, like scikit pipelines. Technically
    TextRemover does not need to be fit and is always in a fitted state
    (any 'is_fitted' checks of an instance will always return True)
    because TextRemover knows everything it needs to know to transform
    data from the parameters. It also has a no-op :meth: `score` method
    to allow dask_ml wrappers.

    Accepts 1D list-like and (possibly ragged) 2D array-likes of strings.
    Accepted 1D containers include python lists, tuples, and sets, numpy
    vectors, pandas series, and polars series. Accepted 2D containers
    include embedded python sequences, numpy arrays, pandas dataframes,
    and polars dataframes. When passed a 1D list-like, returns a python
    list of strings. When passed a 2D array-like, returns a python list
    of python lists of strings. If you pass your data as a dataframe
    with feature names, the feature names are not preserved.

    By definition, a row is removed from 1D data when an entire string
    is removed. This behavior is unavoidable, in this case TextRemover
    must mutate along the example axis. However, the user can control
    this behavior for 2D containers. :param: `remove_empty_rows` is a
    boolean that indicates to TR whether to remove any rows that may
    have become (or may have been given as) empty after removing unwanted
    strings. If True, TR will remove any empty rows from the data and
    those rows will be indicated in the :attr: `row_support_` mask by a
    False in their respective positions. It is possible that empty 1D
    lists are returned. If False, empty rows are not removed from the
    data.

    TextRemover instances that have undergone a transform operation
    expose 2 attributes. :attr: `n_rows_` is the number of rows in the
    data last passed to :meth: `transform`, which may be different than
    the number of rows returned. :attr: `row_support_` is a boolean
    numpy vector indicating which rows were kept (True) and which were
    removed (False) fram the data during the last transform. This mask
    can be applied to a target for the data (if any) so that the rows in
    the target match the rows in the data after transform. The length
    of :attr: `row_support_` must equal :attr: `n_rows_`. Neither of
    these attributes are cumulative, they only reflect the last dataset
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
        regular expression or re.Pattern object is passed, any matching
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
    remove_empty_rows:
        Optional[bool], default=True - whether to remove rows that become
        empty when data is passed in a 2D container. This does not apply
        to 1D data. If True, TR will remove any empty rows from the data
        and that row will be indicated in the :attr: `row_support_` mask
        by a False in that position. If False, empty rows are not removed
        from the data.


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
        RowSupportType - A boolean vector indicating which rows were
        kept in the data during the transform process. Only available if
        a transform has been performed, and only reflects the results of
        the last transform done.


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

    RemoveEmptyRowsType:
        bool

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
        regexp_flags: Optional[RegExpFlagsType] = None,
        remove_empty_rows: Optional[bool] = True
    ) -> None:

        self.str_remove = str_remove
        self.regexp_remove = regexp_remove
        self.regexp_flags = regexp_flags
        self.remove_empty_rows = remove_empty_rows


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
        X:XContainer,
        copy:Optional[bool] = False
    ) -> XWipContainer:

        """
        Remove unwanted strings from the data.


        Parameters
        ----------
        X:
            list-like 1D vector of strings or (possibly ragged) 2D
            array-like of strings - the data.
        copy:
            Optional[bool], default=False - whether to remove unwanted
            strings directly from the original X or from a deepcopy of
            the original X.


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
            self.regexp_flags,
            self.remove_empty_rows
        )

        if copy:
            _X = copy_X(X)
        else:
            _X = X

        _sr = self.str_remove
        _rr = self.regexp_remove
        _rf = self.regexp_flags

        if all(map(isinstance, _X, (str for _ in _X))):
            _X = list(_X)

            self._n_rows = len(_X)

            if _sr is not None:
                _X, self._row_support = _str_1D_core(_X, _sr)
            elif _rr is not None:
                _X, self._row_support = _regexp_1D_core(_X, _rr, _rf)
            else:
                raise Exception
        else:
            # must be 2D -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
            if isinstance(_X, pd.DataFrame):
                _X = list(map(list, _X.values))
            elif isinstance(_X, pl.DataFrame):
                _X = list(map(list, _X.rows()))
            else:
                _X = list(map(list, _X))

            for _row_idx in range(len(_X)):

                if _sr is not None:

                    if isinstance(_sr, list) and _sr[_row_idx] is False:
                        continue

                    # notice the indexer, only need the _X component
                    _X[_row_idx] = _str_1D_core(
                        _X[_row_idx],
                        _sr[_row_idx] if isinstance(_sr, list) else _sr
                    )[0]

                elif _rr is not None:

                    # if rf is a list, that entry must also be False
                    if isinstance(_rr, list) and _rr[_row_idx] is False:
                        continue

                    # notice the indexer, only need the _X component
                    _X[_row_idx] = _regexp_1D_core(
                        _X[_row_idx],
                        _rr[_row_idx] if isinstance(_rr, list) else _rr,
                        _rf[_row_idx] if isinstance(_rf, list) else _rf
                    )[0]
                else:
                    raise Exception

            self._n_rows = len(_X)

            self._row_support = np.ones(self._n_rows, dtype=bool)
            if self.remove_empty_rows:
                for _row_idx in range(self._n_rows-1, -1, -1):
                    if len(_X[_row_idx]) == 0:
                        _X.pop(_row_idx)
                        self._row_support[_row_idx] = False
            # END recursion for 2D -- -- -- -- -- -- -- -- -- -- -- --

        del _sr, _rr, _rf

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



