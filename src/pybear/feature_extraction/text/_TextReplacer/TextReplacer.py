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
    Search 1D vectors or (possibly ragged) 2D arrays for character
    substrings via str.replace or re.sub and make one-to-one replacements.

    TextReplacer (TR) has 2 modes for making replacements, exact string
    matching and regular expressions. Exact string matching uses the
    built-in str.replace method and therefore is bound to the constraints
    of that method. Regular expression mode uses the built-in re.sub
    function.

    The incremental benefit of TR over the 2 foundational functions is
    that you can quickly apply multiple replacement criteria over the
    entire text body. Also, should you need granular control of
    replacements at the individual row level, TR allows customized
    search and replacement criteria for each row in your data.

    For those who don't know how to write regular expressions or don't
    want to spend the time fine-tuning the cryptic patterns, TR's exact
    string matching mode provides quick access to search and replace
    functionality with multiple criteria. str.replace mode is always
    case-sensitive.

    TR regular expression search and replace has the full functionality
    of re.sub. You can pass your search patterns as strings or re.Pattern
    objects. You can pass a callable as the replacement value for your
    search. For example, a TR regexp replacement criteria might be
    ('[a-m]', your_callable()). If you need to use flags, they can be
    passed inside a re.compile object as a search criteria, or as an
    argument to the re.sub function (but not both!) For example, a
    search criteria for the 'regexp_replace' parameter might be
    (re.compile('a', re.I), '', 0). In this case, the re.I flag will
    make your search case agnostic.

    When building search / replace criteria for either string mode
    (str.replace) or regexp mode (re.sub) that will be passed to the
    'str_replace' and 'regexp_replace' parameters, respectively, build
    the signature of the target function inside a tuple. For example,
    the minimum arguments required for str.replace are 'old' and 'new',
    and can take an optional 'count' argument. For the former case, pass
    your arguments in a tuple as ('old string', 'new string'), where
    str.replace will use its default value for count. For the latter
    case, pass your arguments as ('old string', 'new string', count).
    Apply the same logic to the signature of re.sub to pass tuples of
    arguments to 'regexp_replace'. Other than the target string itself,
    re.sub takes 2 required arguments and up to 2 other optional
    arguments. Therefore, TR's 'regexp_replace' parameter will accept
    tuples of 2, 3, or 4 items. The tuples must be built in the same
    order as the signature of the target function. To pass a non-default
    value for the last argument, you must pass all the arguments. When
    working with 2D data, the 'count' arguments for both 'str_replace'
    and 'regexp_replace' apply to each string in the line, not to the
    whole line.

    You can enter multiple tuples of arguments for both 'str_replace'
    and 'regexp_replace' to quickly execute multiple search criteria on
    your data. Pass the tuples described above to python sets. For
    example, multiple search criteria for string mode might be
    {(',', ''), ('.', '', 1), (';' '')}. Similarly, multiple criteria
    for regexp mode might be {('[a-m]', 'A', 0, re.I), ('@', '', 0)}.
    TR works from left to right through the set when searching and
    replacing. Knowing this, the replacements can be gamed so that later
    searches are dependent on earlier replacements.

    To make fine-grained replacements on the individual rows of your
    data, you can pass a python list of the above-described tuples and
    sets. The length of the list must match the length of the data, for
    both 1D and 2D data. To turn off search/replace for particular rows
    of the data, put False in those indices in the list. So your list
    can take tuples of arguments, sets of tuples of arguments, and
    literal False.

    TR can be instantiated with the default parameters, but this will
    result in a no-op. To actually make replacements, the user must set
    at least one or both of 'str_replace' or 'regexp_replace'.

    You can pass search/replace criteria to both string mode and regexp
    mode simultaneously, if you wish. TR does the regexp replacements
    first, so know that whatever string searches are to be done will see
    the text as it was left after the regexp replacements.

    TR does not remove any strings completely. It can leave empty
    strings. Use pybear TextRemover to remove them, if needed.

    TR is a scikit-style transformer with partial_fit, fit, transform,
    fit_transform, set_params, get_params, and score methods. TR is
    technically always fit because it does need to learn anything from
    data to do transformations; it already knows everything it needs to
    know from the parameters. Checks for fittedness will always return
    True, The partial_fit, fit, and score methods are no-ops that allow
    TR to be incorporated into larger workflows such as scikit pipelines
    or dask_ml wrappers. The get_params, set_params, transform, and
    fit_transform methods are fully functional.

    TR accepts 1D list-like vectors of strings or (possibly ragged) 2D
    array-likes of strings. It does not accept pandas dataframes; convert
    your dataframes to numpy arrays or a python list of lists before
    passing to TR. When passed a 1D list-like, a python list of the same
    size is returned. When passed a possibly ragged 2D array-like, an
    identically-shaped list of python lists is returned.


    Type Aliases
    ------------
    XContainer:
        Union[Sequence[str], Sequence[Sequence[str]]]

    ReplaceType:
        Callable[[str, str, Optional[numbers.Integral]], str]

    OldType:
        str

    NewType:
        str

    CountType:
        numbers.Integral

    StrReplaceArgsType:
        Union[
            tuple[OldType, NewType],
            tuple[OldType, NewType, CountType]
        ]

    TRStrReplaceArgsType:
        Union[StrReplaceArgsType, set[StrReplaceArgsType]]

    StrReplaceType:
        Union[
            TRStrReplaceArgsType,
            list[Union[TRStrReplaceArgsType, Literal[False]]],
            None
        ]

    PatternType:
        Callable[[str, Optional[numbers.Integral]], re.Pattern]

    SearchType:
        Union[str, PatternType]

    ReplType:
        Union[str, Callable[[re.Match], str]]

    CountType:
        numbers.Integral

    FlagsType:
        numbers.Integral

    ReSubType:
        Callable[
            [SearchType, ReplType, str, Optional[CountType], Optional[FlagsType]],
            str
        ]

    RegExpReplaceArgsType:
        Union[
            tuple[SearchType, ReplType],
            tuple[SearchType, ReplType, CountType],
            tuple[SearchType, ReplType, CountType, FlagsType],
        ]

    TRRegExpReplaceArgsType:
        Union[RegExpReplaceArgsType, set[RegExpReplaceArgsType]]

    RegExpReplaceType:
        Union[
            TRRegExpReplaceArgsType,
            list[Union[TRRegExpReplaceArgsType, Literal[False]]],
            None
        ]


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









