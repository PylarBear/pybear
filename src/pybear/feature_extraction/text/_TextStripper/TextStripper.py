# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Optional
from typing_extensions import Self, Union
from ._type_aliases import (
    XContainer,
    XWipContainer
)


from pybear.base import (
    FitTransformMixin,
    GetParamsMixin,
    ReprMixin,
    SetParamsMixin,
    check_is_fitted
)

from ._transform import _transform

from ..__shared._validation._1D_2D_X import _val_1D_2D_X
from ..__shared._transform._map_X_to_list import _map_X_to_list

from ....base._copy_X import copy_X



class TextStripper(
    FitTransformMixin,
    GetParamsMixin,
    ReprMixin,
    SetParamsMixin
):

    """
    Strip leading and trailing spaces from 1D or 2D text data. The data
    can only contain strings.

    TextStripper is a scikit-style transformer that has partial_fit, fit,
    transform, fit_transform, set_params, get_params, and score methods.

    TextStripper technically does not need fitting as it already has
    all the information it needs to perform transforms. Checks for
    fittedness will always return True. The partial_fit, fit, and score
    methods are no-ops that allow TextStripper to be incorporated into
    larger workflows such as scikit pipelines or dask_ml wrappers. The
    get_params, set_params, transform, and fit_transform methods are
    fully functional, but get_params and set_params are trivial because
    TextStripper has no parameters and no attributes.

    TextStripper can transform 1D list-likes of strings and (possibly
    ragged) 2D array-likes of strings. Accepted 1D containers include
    python lists, tuples, and sets, numpy vectors, pandas series, and
    polars series. Accepted 2D containers include embedded python
    sequences, numpy arrays, pandas dataframes, and polar dataframes.
    When passed a 1D list-like, a single python list of strings is
    returned. When passed a possibly ragged 2D array-like of strings,
    TextStripper will return an equally sized and also possibly ragged
    python list of python lists of strings.

    TextStripper has no parameters and no attributes.


    Notes
    -----
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


    Examples
    --------
    >>> from pybear.feature_extraction.text import TextStripper as TS
    >>> trfm = TS()
    >>> X = ['  a   ', 'b', '   c', 'd   ']
    >>> trfm.fit_transform(X)
    ['a', 'b', 'c', 'd']
    >>> X = [['w   ', '', 'x   '], ['  y  ', 'z   ']]
    >>> trfm.fit_transform(X)
    [['w', '', 'x'], ['y', 'z']]


    """


    def __init__(self):
        """Initialize the TextStripper instance."""

        pass


    def __pybear_is_fitted__(self):
        return True


    def get_metadata_routing(self):
        raise NotImplementedError(
            f"'get_metadata_routing' is not implemented in TextStripper"
        )


    def partial_fit(
        self,
        X: XContainer,
        y: Optional[Union[any, None]] = None
    ) -> Self:

        """
        No-op batch-wise fit.


        Parameters
        ----------
        X:
            XContainer - the data whose text will be stripped of leading
            and trailing spaces. Ignored.
        y:
            Optional[Union[any, None]], default=None - the target for
            the data. Always ignored.


        Returns
        -------
        -
            self - the TextStripper instance.

        """


        return self


    def fit(
        self,
        X: XContainer,
        y: Optional[Union[any, None]] = None
    ) -> Self:

        """
        No-op one-shot fit.


        Parameters
        ----------
        X:
            XContainer - the data whose text will be stripped of leading
            and trailing spaces. Ignored.
        y:
            Optional[Union[any, None]], default=None - the target for
            the data. Always ignored.


        Returns
        -------
        -
            self - the TextStripper instance.

        """


        return self.partial_fit(X, y)


    def transform(
        self,
        X:XContainer,
        copy:Optional[bool] = False
    ) -> XWipContainer:

        """
        Remove the leading and trailing spaces from 1D or 2D text data.


        Parameters
        ----------
        X:
            XContainer - the data whose text will be stripped of leading
            and trailing spaces.
        copy:
            Optional[bool], default=False - whether to strip the text
            in the original X object or a deepcopy of X.


        Returns
        -------
        -
            XWipContainer - the data with stripped text.


        """

        check_is_fitted(self)

        _val_1D_2D_X(X, _require_all_finite=False)

        if copy:
            _X = copy_X(X)
        else:
            _X = X


        _X: XWipContainer = _map_X_to_list(_X)


        return _transform(_X)


    def score(
        self,
        X: XContainer,
        y: Optional[Union[any, None]] = None
    ) -> None:

        """
        No-op score method.


        Parameters
        ----------
        X:
            XContainer - the data. Ignored.
        y:
            Optional[Union[any, None]], default=None - the target for
            the data. Always ignored.


        Returns
        -------
        -
            None

        """


        check_is_fitted(self)

        return








