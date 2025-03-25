# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Callable, Optional, Sequence
from typing_extensions import Self, Union
from ._type_aliases import XContainer

import re

import pandas as pd
import polars as pl

from ._validation._validation import _validation
from ._transform._transform import _transform

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
    Join adjacent "words" into an N-gram unit, to be handled as a single
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
        *,
        ngrams: Sequence[Sequence[Union[str, re.Pattern]]],
        ngcallable: Optional[Callable[[Sequence[str]], str]],
        sep: Optional[str]
    ) -> None:

        """Initialize the NGramMerger instance."""

        self.ngrams = ngrams
        self.ngcallable = ngcallable
        self.sep = sep


    def __pybear_is_fitted__(self):
        return True


    # def get_params
    # handled by GetParamsMixin


    # def set_params
    # handled by SetParamsMixin


    # def fit_transform
    # handled by FitTransformMixin


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
        Merge N-grams in a (possibly ragged) 2D array-like of strings.


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
            list[list[str]] - the data with all matching n-gram patterns
            replaced with contiguous strings.


        """

        check_is_fitted(self)

        _validation(X, self.ngrams, self.ngcallable, self.sep)

        if copy:
            _X = copy_X(X)
        else:
            _X = X

        # we know from validation it is legit 1D or 2D, do the easy check
        # if all(map(isinstance, _X, (str for _ in _X))):
        #     # then is 1D:
        #     _X = list(_X)
        # else:
        # # then could only be 2D, need to convert to 1D
        if isinstance(_X, pd.DataFrame):
            _X = list(map(list, _X.values))
        elif isinstance(_X, pl.DataFrame):
            _X = list(map(list, _X.rows()))
        else:
            _X = list(map(list, _X))


        return _transform(_X, self.ngrams, self.ngcallable, self.sep)


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





