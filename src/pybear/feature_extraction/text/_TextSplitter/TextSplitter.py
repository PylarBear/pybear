# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Optional, Sequence
from typing_extensions import Self, Union
from ._type_aliases import XContainer

import numbers
from copy import deepcopy

from ....base import (
    FitTransformMixin,
    GetParamsMixin,
    ReprMixin,
    SetParamsMixin
)

from ._validation._validation import _validation
from ._transform._transform import _transform


class TextSplitter(
    FitTransformMixin,
    GetParamsMixin,
    ReprMixin,
    SetParamsMixin
):


    def __init__(
        self,
        *,
        sep: Union[str, Sequence[str], None] = None,
        regexp: Union[str, Sequence[str], None] = None,
        maxsplit: Union[numbers.Integral, Sequence[numbers.Integral], None] = None,
        flags: Union[numbers.Integral, None] = None
    ):

        """
        Split all the strings on the given separator(s).


        Parameters
        ----------
        sep:
            Union[str, Sequence[str], None], default=None - the
            separator(s) to split the strings in X on.
        regexp:
            Union[str, Sequence[str], None], default=None - if using
            regular expressions, the regexp pattern(s) to split the
            strings in X on.
        maxsplit:
            Union[numbers.Integral, Sequence[numbers.Integral], None],
            default=None - the maximum number of splits to perform. If
            passed as a sequence of integers, the number of entries must
            match the number of strings in X.
        flags:
            Union[numbers.Integral, Sequence[numbers.Integral], None] -
            the flags parameter for re.split, if regular expressions are
            being used.


        Notes
        -----
        see python str.split()


        """

        self.sep = sep
        self.regexp = regexp
        self.maxsplit = maxsplit
        self.flags = flags


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

        _validation(X, self.regexp, self.sep, self.maxsplit, self.flags)


        if copy:
            if isinstance(X, (list, set, tuple)):
                _X = deepcopy(X)
            else:
                _X = X.copy()
        else:
            _X = X


        return _transform(_X, self.regexp, self.sep, self.maxsplit, self.flags)



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




