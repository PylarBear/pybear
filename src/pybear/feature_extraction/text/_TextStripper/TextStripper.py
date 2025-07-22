# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Optional
from typing_extensions import (
    Any,
    Self
)
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
    """Strip leading and trailing spaces from 1D or 2D text data.

    The data can only contain strings.

    `TextStripper` is a scikit-style transformer that has `partial_fit`,
    `fit`, `transform`, `fit_transform`, `set_params`, `get_params`, and
    `score` methods.

    `TextStripper` technically does not need fitting as it already has
    all the information it needs to perform transforms. Checks for
    fittedness will always return True. The `partial_fit`, `fit`, and
    `score` methods are no-ops that allow `TextStripper` to be
    incorporated into larger workflows such as scikit pipelines or
    dask_ml wrappers. The `get_params`, `set_params`, `transform`, and
    `fit_transform` methods are fully functional, but `get_params` and
    `set_params` are trivial because `TextStripper` has no parameters
    and no attributes.

    `TextStripper` can transform 1D list-likes of strings and (possibly
    ragged) 2D array-likes of strings. Accepted 1D containers include
    Python lists, tuples, and sets, numpy vectors, pandas series, and
    polars series. Accepted 2D containers include embedded Python
    sequences, numpy arrays, pandas dataframes, and polar dataframes.
    When passed a 1D list-like, a single Python list of strings is
    returned. When passed a possibly ragged 2D array-like of strings,
    TextStripper will return an equally sized and also possibly ragged
    Python list of Python lists of strings.

    `TextStripper` has no parameters and no attributes.

    Notes
    -----

    **Type Aliases**

    PythonTypes:
        Union[Sequence[str], Sequence[Sequence[str]]]

    NumpyTypes:
        numpy.ndarray[str]

    PandasTypes:
        Union[panda.core.series.Series, pandas.core.frame.DataFrame]

    PolarsTypes:
        Union[polars.series.Series, polars.dataframe.DataFrame]

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


    def __pybear_is_fitted__(self) -> bool:
        return True


    def get_metadata_routing(self):
        """'get_metadata_routing' is not implemented in TextStripper."""
        raise NotImplementedError(
            f"'get_metadata_routing' is not implemented in TextStripper."
        )


    def partial_fit(
        self,
        X: XContainer,
        y: Optional[Any] = None
    ) -> Self:
        """No-op batch-wise fit.

        Parameters
        ----------
        X : XContainer
            The data whose text will be stripped of leading and trailing
            spaces. Ignored.
        y : Optional[Any], default=None
            The target for the data. Always ignored.

        Returns
        -------
        self : object
            The `TextStripper` instance.

        """

        return self


    def fit(
        self,
        X: XContainer,
        y: Optional[Any] = None
    ) -> Self:
        """No-op one-shot fit.

        Parameters
        ----------
        X : XContainer
            The data whose text will be stripped of leading and trailing
            spaces. Ignored.
        y : Optional[Any], default=None
            The target for the data. Always ignored.

        Returns
        -------
        self : object
            The `TextStripper` instance.

        """

        return self.partial_fit(X, y)


    def transform(
        self,
        X:XContainer,
        copy:Optional[bool] = False
    ) -> XWipContainer:
        """Remove the leading and trailing spaces from 1D or 2D text data.

        Parameters
        ----------
        X : XContainer
            The data whose text will be stripped of leading and trailing
            spaces.
        copy : Optional[bool], default=False
            Whether to strip the text in the original `X` object or a
            deepcopy of `X`.

        Returns
        -------
        X_tr: XWipContainer
            The data with stripped text.

        """

        check_is_fitted(self)

        _val_1D_2D_X(X, _require_all_finite=False)

        if copy:
            X_tr = copy_X(X)
        else:
            X_tr = X


        X_tr: XWipContainer = _map_X_to_list(X_tr)


        return _transform(X_tr)


    def score(
        self,
        X: Any,
        y: Optional[Any] = None
    ) -> None:
        """No-op score method.

        Parameters
        ----------
        X : Any
            The data. Ignored.
        y : Optional[Any], default=None
            The target for the data. Ignored.

        Returns
        -------
        None

        """


        check_is_fitted(self)

        return




