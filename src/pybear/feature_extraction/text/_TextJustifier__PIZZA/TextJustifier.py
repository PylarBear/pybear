# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Optional
from typing_extensions import Self, Union
from ._type_aliases import XContainer

from copy import deepcopy

from ....base import (
    FitTransformMixin,
    GetParamsMixin,
    ReprMixin,
    SetParamsMixin,
    check_is_fitted
)




class TextJustifier(
    FitTransformMixin,
    GetParamsMixin,
    ReprMixin,
    SetParamsMixin
):

    """


    Parameters
    ----------


    """

    def __init__(
        self,
        n_characters=79,
        sep=' '
    ):

        """Initialize the TextJustifier instance."""

        self.n_characters = n_characters
        self.sep=' '


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
            Union[Sequence[str], Sequence[Sequence[str]]] - The data to
            justify. Always ignored.
        y:
            Optional[Union[any, None]], default=None - the target for
            the data. Always ignored.


        Return
        ------
        -
            self - the TextJustifier instance.


        """

        check_is_fitted(self)

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
            Union[Sequence[str], Sequence[Sequence[str]]] - The data to
            justify. Always ignored.
        y:
            Optional[Union[any, None]], default=None - the target for
            the data. Always ignored.


        Return
        ------
        -
            self - the TextJustifier instance.


        """

        check_is_fitted(self)

        return self.partial_fit(X, y)


    def transform(
        self,
        X: XContainer,
        copy: Optional[bool] = True
    ) -> XContainer:

        """
        Justify the text in a 1D list-like of strings or a (possibly
        ragged) 2D array like of strings.


        Parameters
        ----------
        X:
            Union[Sequence[str], Sequence[Sequence[str]]] - The data to
            justify.
        copy:
            Optional[bool], default=True - whether to directly operate
            on the passed X or on a copy.


        Return
        ------
        -
            self - the TextJustifier instance.


        """

        check_is_fitted(self)

        # _validation(X)

        if copy:
            if isinstance(X, (list, tuple, set)) or not hasattr(X, 'copy'):
                _X = deepcopy(X)
            else:
                _X = X.copy()
        else:
            _X = X

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
            Union[Sequence[str], Sequence[Sequence[str]]] - The data to
            justify. Always ignored.
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





