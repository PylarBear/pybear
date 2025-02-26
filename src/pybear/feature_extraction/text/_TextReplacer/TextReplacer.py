# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Optional
from typing_extensions import Self, Union
from ._type_aliases import (
    XContainer,
    StrReplaceType,
    RegExpReplaceType
)

from copy import deepcopy

from ....base import (
    FitTransformMixin,
    GetParamsMixin,
    ReprMixin,
    SetParamsMixin,
    check_is_fitted
)

from ._validation._validation import _validation
from ._transform._transform import _transform



class TextReplacer(
    FitTransformMixin,
    GetParamsMixin,
    ReprMixin,
    SetParamsMixin
):

    """
    Search for character substrings via str.replace or re.sub and make a
    one-to-one replacement.

    str.replace mode is subject to the terms of the built-in str.replace
    function. One condition is that these searches are always
    case-sensitive.

    TextReplacer (TR) regular expression search and replace has the full
    functionality of re.sub. You
    can pass a callable as the replacement value for your search, subject
    to the conditions imposed by re.sub. For example, a TR replacement criteria
    might be {'[a-m]': your_callable}.

    TR does not directly accept regular expression flags as a
    parameter. If you need to use them, they can be passed inside a
    re.compile object as a search criteria. For example, a search
    criteria for the 'regexp_replace' parameter might be {'a': ''}. If
    you want to make your search case agnostic, then pass the flag via
    re.compile as the search term, like {re.compile('a', re.I): ''}.

    When working with 2D data, the 'count' parameters for both 'str_replace'
    and 'regexp_replace', count applies to each string in the line, not for
    the whole line.

    TR does not remove any strings completely. It can leave
    empty strings. Use pybear TextRemover to remove them, if needed.

    TR can be instantiated with the default parameters, but this will
    result in a no-op.
    To actually make replacements, the user must set at least one or
    both of 'str_replace' or 'regexp_replace'.

    TR is a scikit-style transformer with partial_fit, fit,
    transform, fit_transform, set_params, get_params, and score methods.
    TR is technically always fit because it does need to learn
    anything from data to do transformations; it already knows everything
    it needs to know from the parameters. The partial_fit, fit, and
    score methods are no-ops that allow TR to be incorporated
    into larger workflows such as scikit pipelines or dask_ml wrappers.
    The get_params, set_params, transform, and fit_transform methods are
    fully functional.

    TR accepts 1D list-like vectors of strings or (possibly
    ragged) 2D array-likes of strings. It does not accept pandas
    dataframes; convert your dataframes to numpy arrays or a python list
    of lists before passing to TR. When passed a 1D list-like,
    a python list of the same size is returned. When passed a possibly
    ragged 2D array-like, a possibly ragged list of python lists is
    returned.


    Type Aliases
    ------------



    Parameters
    ----------
    str_replace:
        StrReplaceType, default=None - the
        character substring(s) to replace by exact text matching and
        their replacement(s). Uses str.replace. Case-sensitive.
    regexp_replace:
        RegExpReplaceType, default=None - the regular expression
        pattern(s) to substitute and their replacement(s). Uses re.sub.


    See Also
    --------
    str.replace
    re.sub


    Examples
    --------


    """


    def __init__(
        self,
        *,
        str_replace: Optional[StrReplaceType] = None,
        regexp_replace: Optional[RegExpReplaceType] = None
    ) -> None:

        """Initialize instance parameters."""

        self.str_replace = str_replace
        self.regexp_replace = regexp_replace


    def __pybear_is_fitted__(self):
        return True


    def get_metadata_routing(self):
        raise NotImplementedError(
            f"'get_metadata_routing' is not implemented in TextReplacer"
        )


    def partial_fit(
        self,
        X: XContainer,
        y: Optional[Union[any, None]] = None
    ) -> Self:

        """
        No-op batch-wise fit of the TextReplacer instance.


        Parameters
        ----------
        X:
            Union[Sequence[str], Sequence[Sequence[str]]] - the data
            that is to undergo search and replace. Always ignored.
        y:
            Optional[Union[any, None]], default = None - the target for
            the data. Always ignored.


        Returns
        -------
        -
            self - the TextReplacer instance.


        """

        check_is_fitted(self)

        return self


    def fit(
        self,
        X: XContainer,
        y: Optional[Union[any, None]] = None
    ) -> Self:

        """
        No-op one-shot batch-wise fit of the TextReplacer instance.


        Parameters
        ----------
        X:
            Union[Sequence[str], Sequence[Sequence[str]]] - the data
            that is to undergo search and replace. Always ignored.
        y:
            Optional[Union[any, None]], default = None - the target for
            the data. Always ignored.


        Returns
        -------
        -
            self - the TextReplacer instance.


        """

        check_is_fitted(self)

        return self.partial_fit(X, y)


    def transform(
        self,
        X: XContainer,
        copy: Optional[bool] = True
    ) -> XContainer:

        """
        Search the data for matches against the search criteria and make
        the specified replacements.


        Parameters
        ----------
        X:
            Union[Sequence[str], Sequence[Sequence[str]]] - the data
            whose strings will be searched and may be replaced in whole
            or in part.
        copy:
            Optional[bool]], default = True - whether to make the
            replacements directly on the given data or a copy of it.


        Returns
        -------
        -
            XContainer: the data with replacements made.


        """


        check_is_fitted(self)

        _validation(X, self.str_replace, self.regexp_replace)

        if copy:
            if isinstance(X, (list, tuple, set)) or not hasattr(X, 'copy'):
                _X = deepcopy(X)
            else:
                _X = X.copy()
        else:
            _X = X

        if all(map(isinstance, _X, (str for _ in _X))):
            _X = list(_X)
        else:
            _X = list(map(list, _X))

        return _transform(_X, self.str_replace, self.regexp_replace)


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
            Union[Sequence[str], Sequence[Sequence[str]]] - the data
            that is to undergo search and replace. Always ignored.
        y:
            Optional[Union[any, None]], default = None - the target for
            the data. Always ignored.


        Returns
        -------
        -
            None


        """

        check_is_fitted(self)

        return









