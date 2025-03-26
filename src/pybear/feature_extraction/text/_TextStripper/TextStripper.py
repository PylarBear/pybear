# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Optional, Sequence
from typing_extensions import Self, Union


from pybear.base import (
    FitTransformMixin,
    GetParamsMixin,
    ReprMixin,
    SetParamsMixin,
    check_is_fitted
)

from ._transform._transform import _transform

from ....base._check_dtype import check_dtype
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
    ragged) 2D array-likes of strings. It cannot take pandas dataframes;
    convert your pandas dataframes to numpy arrays or python lists of
    lists before passing to TextStripper. Pandas and polars series give
    expected results. Polars dataframes do not fail the validation but
    pybear cannot guarantee sensible results. When passed a 1D list-like,
    a single python list of strings is returned. When passed a possibly
    ragged 2D array-like of strings, TextStripper will return an equally
    sized and also possibly ragged python list of python lists of
    strings.

    TextStripper has no parameters and no attributes.


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
        X: Union[Sequence[str], Sequence[Sequence[str]]],
        y: Optional[Union[any, None]] = None
    ) -> Self:

        """
        No-op batch-wise fit.


        Parameters
        ----------
        X:
            Union[Sequence[str], Sequence[Sequence[str]]] - the data
            whose text will be stripped of leading and trailing spaces.
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
        X: Union[Sequence[str], Sequence[Sequence[str]]],
        y: Optional[Union[any, None]] = None
    ) -> Self:

        """
        No-op one-shot fit.


        Parameters
        ----------
        X:
            Union[Sequence[str], Sequence[Sequence[str]]] - the data
            whose text will be stripped of leading and trailing spaces.
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
        X: Union[Sequence[str], Sequence[Sequence[str]]],
        copy: Optional[bool] = True
    ) -> Union[list[str], list[list[str]]]:

        """
        Remove the leading and trailing spaces from 1D or 2D text data.


        Parameters
        ----------
        X:
            Union[Sequence[str], Sequence[Sequence[str]]] - the data
            whose text will be stripped of leading and trailing spaces.
        copy:
            Optional[bool], default=None - whether to strip the text
            in the original X object or a copy of X.


        Returns
        -------
        -
            Union[list[str], list[list[str]]] - the data with stripped
            text.

        """

        check_is_fitted(self)

        # check_dtype takes 1 or 2D
        check_dtype(X, allowed='str', require_all_finite=True)

        if copy:
            _X = copy_X(X)
        else:
            _X = X


        if all(map(isinstance, _X, (str for _ in _X))):
            _X = list(_X)
        else:
            _X = list(map(list, _X))


        return _transform(_X)


    def score(
        self,
        X: Union[Sequence[str], Sequence[Sequence[str]]],
        y: Optional[Union[any, None]] = None
    ) -> Self:

        """
        No-op score method.


        Parameters
        ----------
        X:
            Union[Sequence[str], Sequence[Sequence[str]]] - the data.
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








