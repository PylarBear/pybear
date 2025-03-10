# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Optional
from typing_extensions import Self, Union

from ._type_aliases import (
    XContainer,
    UpperType
)

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



class TextNormalizer(
    FitTransformMixin,
    GetParamsMixin,
    ReprMixin,
    SetParamsMixin
):

    """
    Normalize all text in a dataset to upper-case, lower-case, or leave
    unchanged. The data can only contain strings.

    TextNormalizer (TN) is a scikit-style transformer with partial_fit,
    fit, transform, fit_transform, get_params, set_params, and score
    methods. An instance is always in a 'fitted' state, and checks for
    fittedness will always return True. This is because TN technically
    does not need to be fit; it already knows everything it needs to
    know to do transforms from the single parameter. The partial_fit,
    fit, and score methods are no-op; they exist to fulfill the API and
    to enable TN to be incorporated into workflows such as scikit
    pipelines and dask_ml wrappers.

    TextNormalizer accepts 1D list-like vectors of strings and (possibly
    ragged) 2D array-likes of strings. It does not accept pandas
    dataframes, convert your pandas dataframes to numpy array or python 
    lists before using TextNormalizer. Pandas and polars series give
    expected results. Polars dataframes pass the validation but pybear
    cannot guarantee sensible results.  The returned objects are always
    constructed with python lists, and have shape identical to the shape
    of the inputted data.


    Parameters
    ----------
    upper:
        Union[bool, None] - If True, convert all text in X to upper-case;
        if False, convert to lower-case; if None, do a no-op.

    
    See Also
    --------
    str.lower()
    str.upper()
    
    
    Examples
    --------
    >>> from pybear.feature_extraction.text import TextNormalizer
    >>> trfm = TextNormalizer(upper=False)
    >>> X1 = ['ThE', 'cAt', 'In', 'ThE', 'hAt']
    >>> trfm.fit_transform(X1)
    ['the', 'cat', 'in', 'the', 'hat']
    >>> trfm.set_params(upper=True)
    TextNormalizer()
    >>> X2 = [['One', 'Two', 'Three'], ['Ichi', 'Ni', 'Sa']]
    >>> trfm.fit_transform(X2)
    [['ONE', 'TWO', 'THREE'], ['ICHI', 'NI', 'SA']]
    
    
    """


    def __init__(
        self,
        upper: UpperType = True
    ) -> None:

        self.upper = upper



    def __pybear_is_fitted__(self):
        return True


    def get_metadata_routing(self):
        raise NotImplementedError(
            f"'get_metadata_routing' is not implemented in TextNormalizer"
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
            Union[Sequence[str], Sequence[Sequence[str]]] - the data
            whose text will be normalized.
        y:
            Optional[Union[any, None]], default=None - the target for
            the data. Always ignored.


        Returns
        -------
        -
            self - the TextNormalizer instance.

        """

        check_is_fitted(self)


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
            Union[Sequence[str], Sequence[Sequence[str]]] - the data
            whose text will be normalized.
        y:
            Optional[Union[any, None]], default=None - the target for
            the data. Always ignored.


        Returns
        -------
        -
            self - the TextNormalizer instance.

        """

        check_is_fitted(self)

        return self.partial_fit(X, y)


    def transform(
        self,
        X: XContainer,
        copy: Optional[bool] = True
    ) -> XContainer:

        """
        Normalize the text in a dataset.


        Parameters
        ----------
        X:
            Union[Sequence[str], Sequence[Sequence[str]]] - the data
            whose text will be normalized.
        copy:
            Optional[bool], default=None - whether to normalize the text
            in the original X object or a copy of X.


        Returns
        -------
        -
            Union[list[str], list[list[str]]] - the data with normalized 
            text.

        """

        check_is_fitted(self)

        _validation(X, self.upper)

        if copy:
            _X = copy_X(X)
        else:
            _X = X


        if all(map(isinstance, _X, (str for _ in _X))):
            _X = list(_X)
        else:
            _X = list(map(list, _X))


        return _transform(_X, self.upper)



    def score(
        self,
        X: XContainer,
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















