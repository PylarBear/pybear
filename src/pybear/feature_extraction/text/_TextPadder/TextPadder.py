# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Optional, Sequence
from typing_extensions import Self, Union

from copy import deepcopy



from ....base import (
    FitTransformMixin,
    GetParamsMixin,
    ReprMixin,
    SetParamsMixin,
    check_is_fitted
)


from ._transform._transform import _transform



class TextPadder(
    FitTransformMixin,
    GetParamsMixin,
    ReprMixin,
    SetParamsMixin
):

    def __init__(
        self,
        sep: Optional[Union[str, Sequence[str]], None] = None,  # pizza make a decision about default
        fill: Optional[str] = ''
    ) -> None:

        """
        Pizza
        Map ragged text data to a shaped array.


        Parameters
        ----------
        sep:
            Optional[Union[str, Sequence[str], None]], default=" " -
            pizza make a decision about default
            if
            passing 1D vectors of strings, the character sequence(s) to
            split on.
            Ignored if passing 2D tokenized strings.
        maxsplit:
            Optional[Union[numbers.Integral, Sequence[numbers.Integral]],
            default = -1 - the maximum number of splits to perform per string.
            If a sequence of integers, X must be 1D and the the number of
            entries in maxsplit must match the number of strings in X.
        fill:
            Optional[str], default="" -  The character sequence to pad
            text sequences with.

        """

        self.sep = sep
        self.fill = fill


    # handled by GetParamsMixin
    # def get_params(self, deep:Optional[bool] = True):


    # handled by SetParamsMixin
    # def set_params(self, **params):


    def __pybear_is_fitted__(self):
        # this is always fitted because it doesnt need to be fit!

        # pizza think on this... doesnt need to be fitted, so maybe all
        # this clutter can come out?

        return True



    def partial_fit(
        self,
        X,
        y: Optional[Union[any, None]] = None
    ) -> Self:

        """
        No-op batch-wise fitting operation.


        Parameters
        ----------
        X:
            1D list-like of shape (n_samples,) or 2D array-like of
            (possibly ragged) shape (n_samples, n_features) - The data.
        y:
            Optional[Union[any, None]], default = None. The target for the
            data. Always ignored.


        Return
        ------
        -
            self - the TextPadder instance.

        """


        check_is_fitted(self)

        _validation()


        return self


    def fit(
        self,
        X,
        y: Optional[Union[any, None]] = None
    ) -> Self:

        """
        No-op one-shot fitting operation.


        Parameters
        ----------
        X:
            1D list-like of shape (n_samples,) or 2D array-like of
            (possibly ragged) shape (n_samples, n_features) - The data.
        y:
            Optional[Union[any, None]], default = None. The target for the
            data. Always ignored.


        Return
        ------
        -
            self - the TextPadder instance.

        """

        check_is_fitted(self)

        _validation()


        return self



    def transform(
        self,
        X,
        copy: Optional[bool] = True
    ):

        """
        Map ragged text data to a shaped array.


        Parameters
        ----------
        X:
            1D list-like of shape (n_samples,) or 2D array-like of
            (possibly ragged) shape (n_samples, n_features) - The data
            to be transformed.
        copy:
            Optional[bool], default=True - whether to make a copy of the
            data before performing the transformation.


        Return
        ------
        -
            self - the TextPadder instance.

        """


        check_is_fitted(self)

        _validation()

        if copy:
            if isinstance(X, (list, set, tuple)):
                _X = deepcopy(X)
            else:
                _X = X.copy()
        else: _X = X


        return _transform(_X)


    # handled by FitTransformMixin
    # def fit_transform(self, X):


    def score(
        self,
        X,
        y: Optional[Union[any, None]] = None
    ) -> None:

        """
        No-op score method.


        Parameters
        ----------
        X:
            The data.
        y:
            Optional[Union[any, None]], default = None - the target for
            the data.

        Return
        ------
        -
            None


        """


        check_is_fitted(self)


        return





