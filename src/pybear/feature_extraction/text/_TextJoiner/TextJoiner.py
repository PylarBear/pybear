# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Sequence, Optional
from typing_extensions import Self, Union
from ._type_aliases import (
    XContainer,
    OutputContainer
)

from ._validation._validation import _validation
from ._transform._condition_sep import _condition_sep
from ._transform._transform import _transform

from ....base import (
    FitTransformMixin,
    GetParamsMixin,
    ReprMixin,
    SetParamsMixin,
    check_is_fitted
)

from ....base._copy_X import copy_X



class TextJoiner(
    FitTransformMixin,
    GetParamsMixin,
    ReprMixin,
    SetParamsMixin
):

    """
    Join a (possibly ragged) 2D array-like of (perhaps tokenized) strings
    across rows with the 'sep' character string(s).

    When passed a 2D array-like of strings, TextJoiner joins each
    row-wise sequence of strings on the value given by :param: 'sep' and
    returns a 1D python list of joined strings in place of the original
    inner containers.

    The 'sep' parameter can be passed as a single character string, in
    which case all strings in the data will be joined by that string.
    'sep' can also be passed as a 1D sequence of strings, whose length
    must equal the number of rows of text in the data. In that case,
    TextJoiner uses the string in each position of the 1D sequence to
    join the corresponding row of text in the data.

    TextJoiner is a full-fledged scikit-style transformer. It has fully
    functional get_params, set_params, transform, and fit_transform
    methods. It also has partial_fit, fit, and score methods, which are
    no-ops. TextJoiner technically does not need to be fit because it
    alreay knows everything it needs to do transformation from the 'sep'
    parameter. These no-op methods are available to fulfill the scikit
    transformer API and make TextJoiner suitable for incorporation into
    larger workflows, such as Pipelines and dask_ml wrappers.


    Parameters
    ----------
    _sep:
        Optional[Union[str, Sequence[str]]], default=' ' - The character
        sequence to insert between individual strings when joining the
        2D input data across rows. If a 1D sequence of strings, then the
        'sep' value in each position is used to join the corresponding
        row in X.


    Notes
    -----
    Type Aliases

    PythonTypes: Sequence[Sequence[str]]

    NumpyTypes: npt.NDArray[str]

    PandasTypes: pd.DataFrame

    PolarsTypes: pl.DataFrame

    XContainer: Union[PythonTypes, NumpyTypes, PandasTypes, PolarsTypes]

    OutputContainer: list[str]


    """


    def __init__(
        self,
        *,
        sep: Optional[Union[str, Sequence[str]]] = ' '
    ) -> None:
        """Initialize the TextJoiner instance."""

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
        No-op batch-wise fit method.


        Parameters
        ----------
        X:
            XContainer - the (possibly ragged) 2D container of text to
            be joined along rows using the 'sep' character string(s).
            Ignored.
        y:
            Optional[Union[any, None]], default=None - the target for
            the data. Always ignored.


        Return
        ------
        -
            self: the TextJoiner instance.


        """

        return self


    def fit(
        self,
        X: XContainer,
        y: Optional[Union[any, None]] = None
    ) -> Self:

        """
        No-op one-shot fit method.


        Parameters
        ----------
        X:
            XContainer - the (possibly ragged) 2D container of text
            to be joined along rows using the 'sep' character string(s).
            Ignored.
        y:
            Optional[Union[any, None]], default=None - the target for
            the data. Always ignored.


        Return
        ------
        -
            self: the TextJoiner instance.


        """

        return self.partial_fit(X, y)


    def transform(
        self,
        X: XContainer,
        copy: Optional[bool] = True
    ) -> OutputContainer:

        """
        Convert each row of strings in X to a single string, joining on
        the string character sequence(s) provided by the 'sep' parameter.
        Returns a python list of strings.


        Parameters
        ----------
        X:
            XContainer - the (possibly ragged) 2D container of text
            to be joined along rows using the 'sep' character string(s).


        Return
        ------
        -
            OutputContainer - A single list containing strings, one
            string for each row in the original X.

        """

        check_is_fitted(self)

        _validation(X, self.sep)

        if copy:
            _X = copy_X(X)
        else:
            _X = X

        _X = list(map(list, _X))

        _sep = _condition_sep(self.sep, len(X))

        return _transform(_X, _sep)


    def score(
        self,
        X: XContainer,
        y: Optional[Union[any, None]] = None
    ) -> Self:

        """
        No-op one-shot score method. Needs to be here for dask_ml
        wrappers.


        Parameters
        ----------
        X:
            XContainer - the (possibly ragged) 2D container of text
            to be joined along rows using the 'sep' character string(s).
            Ignored.
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














