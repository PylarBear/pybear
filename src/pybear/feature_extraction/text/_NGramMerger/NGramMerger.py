# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Optional
from typing_extensions import Self, Union
from ._type_aliases import (
    XContainer,
    StrNGramHandlerType,
    RegExpNGramHandlerType,
    StrSepType,
    RegExpSepType
)

import pandas as pd
import polars as pl

from ....base import (
    FitTransformMixin,
    GetParamsMixin,
    ReprMixin,
    SetParamsMixin,
    check_is_fitted
)

from ....base._copy_X import copy_X



class NGramMerger(
    FitTransformMixin,
    GetParamsMixin,
    ReprMixin,
    SetParamsMixin
):

    """
    Join adjacent "words" into an N-gram unit, to be handle as a single
    "word".




    Parameters
    ----------


    Notes
    -----
    Type Aliases


    Examples
    --------



    """


    def __init__(
        self,
        regexp_mode: Optional[bool],
        ngram_handler: Optional[Union[StrNGramHandlerType, RegExpNGramHandlerType]],
        sep: Optional[Union[StrSepType, RegExpSepType]]
    ):

        """Initialize the NGramMerger instance."""

        self.regexp_mode = regexp_mode
        self.ngram_handler = ngram_handler
        self.sep = sep


    def __pybear_is_fitted__(self):
        return True


    def partial_fit(
        self,
        X: XContainer,
        y: Optional[Union[any, None]] = None
    ) -> Self:

        """
        No-op batch-wise fit operation.


        Parameters
        ----------
        X:
            XContainer - The data. Ignored.
        y:
            Optional[Union[any, None]], default=None - the target for
            the data. Always ignored.


        Return
        ------
        -
            self - the NGramMerger instance.


        """


        return self


    def fit(
        self,
        X: XContainer,
        y: Optional[Union[any, None]] = None
    ) -> Self:

        """
        No-op one-shot fit operation.


        Parameters
        ----------
        X:
            XContainer - The data. Ignored.
        y:
            Optional[Union[any, None]], default=None - the target for
            the data. Always ignored.


        Return
        ------
        -
            self - the NGramMerger instance.


        """


        return self.partial_fit(X, y)


    def transform(
        self,
        X: XContainer,
        copy: Optional[bool] = True
    ) -> XContainer:

        """
        Merge N-grams in a 1D list-like of strings or a (possibly
        ragged) 2D array like of strings.


        Parameters
        ----------
        X:
            XContainer - The data.
        copy:
            Optional[bool], default=True - whether to directly operate
            on the passed X or on a copy.


        Return
        ------
        -
            self - the NGramMerger instance.


        """

        check_is_fitted(self)

        # _validation(X)

        if copy:
            _X = copy_X(X)
        else:
            _X = X

        # we know from validation it is legit 1D or 2D, do the easy check
        if all(map(isinstance, _X, (str for _ in _X))):
            # then is 1D:
            _X = list(_X)
        else:
            # then could only be 2D, need to convert to 1D
            if isinstance(_X, pd.DataFrame):
                _X = list(map(list, _X.values))
            elif isinstance(_X, pl.DataFrame):
                _X = list(map(list, _X.rows()))
            else:
                _X = list(map(list, _X))

        # _transform(X, ...)

        return _X


    def score(
        self,
        X: XContainer,
        y: Optional[Union[any, None]] = None
    ) -> None:

        """
        No-op score method. Needs to be here for dask_ml wrappers.


        Parameters
        ----------
        X:
            XContainer - The data. Ignored.
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





