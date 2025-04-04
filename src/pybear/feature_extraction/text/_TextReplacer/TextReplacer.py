# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Optional
from typing_extensions import Self, Union
from ._type_aliases import (
    XContainer,
    XWipContainer,
    ReplaceType,
    WipReplaceType,
    CaseSensitiveType,
    FlagsType
)

from ._validation._validation import _validation
from ._transform._special_param_conditioner import _special_param_conditioner
from ._transform._regexp_1D_core import _regexp_1D_core

from ....base import (
    FitTransformMixin,
    GetParamsMixin,
    ReprMixin,
    SetParamsMixin,
    check_is_fitted
)

from ..__shared._transform._map_X_to_list import _map_X_to_list

from ....base._copy_X import copy_X



class TextReplacer(
    FitTransformMixin,
    GetParamsMixin,
    ReprMixin,
    SetParamsMixin
):

    """
    Search 1D vectors or (possibly ragged) 2D arrays for character
    substrings and make one-to-one replacements.

    TextReplacer (TR)

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
    search criteria for the :param: `regexp_replace` parameter might be
    (re.compile('a', re.I), '', 0). In this case, the re.I flag will
    make your search case agnostic.

    The search / replace criteria that you enter for :param: `str_replace`
    (string mode) and :param: `regexp_replace` (regexp mode) will be
    passed directly to str.replace and re.sub, respectively. To construct
    input for these parameters, build the signature of the target
    function inside a tuple. For example, the minimum arguments required
    for str.replace are 'old' and 'new', and can take an optional 'count'
    argument. For the former case, pass your arguments in a tuple as
    ('old string', 'new string'), where str.replace will use its default
    value for count. For the latter case, pass your arguments as
    ('old string', 'new string', count). Apply the same logic to the
    signature of re.sub to pass arguments to :param: `regexp_replace`.
    Other than the target string itself, re.sub takes 2 required
    arguments and up to 2 other optional arguments. Therefore, you can
    pass tuples of 2, 3, or 4 items to :param: `regexp_replace`. The
    tuples must be built in the same order as the signature of the target
    function. To pass a non-default value for the last argument, you must
    pass all the arguments. When working with 2D data, the 'count'
    arguments for both :param: `str_replace` and :param: `regexp_replace`
    apply to each string in the line, not to the whole line.

    You can enter multiple tuples of arguments for :param: `str_replace`
    and :param: `regexp_replace` to quickly execute multiple search
    criteria on your data. Pass the tuples described above to python
    sets. For example, multiple search criteria for string mode might be
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
    result in a no-op. To actually make replacements, you must set at
    least one of :param: `str_replace` or :param: `regexp_replace`.

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
    True. The partial_fit, fit, and score methods are no-ops that allow
    TR to be incorporated into larger workflows such as scikit pipelines
    or dask_ml wrappers. The get_params, set_params, transform, and
    fit_transform methods are fully functional.

    TR accepts 1D list-like vectors of strings or (possibly ragged) 2D
    array-likes of strings. Accepted 1D containers include python lists,
    tuples, and sets, numpy vectors, pandas series, and polar series.
    Accepted 2D objects include python embedded sequences of sequences,
    numpy arrays, pandas dataframes, and polars dataframe, When passed a
    1D list-like, a python list of the same size is returned. When
    passed a possibly ragged 2D array-like, an identically-shaped list
    of python lists is returned.


    Parameters
    ----------
    str_replace:
        StrReplaceType, default=None - the character substring(s) to
        replace by exact text matching and their replacement(s). Uses
        str.replace. Case-sensitive.
    regexp_replace:
        RegExpReplaceType, default=None - the regular expression
        pattern(s) to substitute and their replacement(s). Uses re.sub.


    Notes
    -----
    Type Aliases

    PythonTypes:
        Union[Sequence[str], Sequence[Sequence[str]], set[str]]

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

    FindType:
        Union[str, re.Pattern[str]]
    SubstituteType:
        Union[str, Callable[[str], str]]
    PairType:
        tuple[FindType, SubstituteType]
    ReplaceSubType:
        Union[None, PairType, tuple[PairType, ...]]
    ReplaceType:
        Optional[Union[ReplaceSubType, list[ReplaceSubType]]]

    WipPairType:
        tuple[re.Pattern[str], SubstituteType]
    WipReplaceSubType:
        Union[None, WipPairType, tuple[WipPairType, ...]]
    WipReplaceType:
        Optional[Union[WipReplaceSubType, list[WipReplaceSubType]]]

    CaseSensitiveType:
        Optional[Union[bool, list[Union[bool, None]]]]

    FlagType:
        Union[None, numbers.Integral]
    FlagsType:
        Optional[Union[FlagType, list[FlagType]]]


    See Also
    --------
    str.replace
    re.sub


    Examples
    --------
    >>> from pybear.feature_extraction.text import TextReplacer as TR
    >>> trfm = TR(str_replace={(',', ''),('.', '')})
    >>> X = ['To be, or not to be, that is the question.']
    >>> trfm.fit_transform(X)
    ['To be or not to be that is the question']
    >>> trfm.set_params(replace=('b', ''))
    TextReplacer(replace=('b', ''))
    >>> trfm.fit_transform(X)
    ['To e, or not to e, that is the question.']


    """


    def __init__(
        self,
        *,
        replace: Optional[ReplaceType] = None,
        case_sensitive: CaseSensitiveType = True,
        flags: FlagsType = None
    ) -> None:

        """Initialize the TextReplacer instance."""

        self.replace = replace
        self.case_sensitive = case_sensitive
        self.flags = flags


    def __pybear_is_fitted__(self):
        return True


    def get_metadata_routing(self):
        raise NotImplementedError(
            f"'get_metadata_routing' is not implemented in TextReplacer"
        )


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
        No-op batch-wise fit of the TextReplacer instance.


        Parameters
        ----------
        X:
            XContainer - 1D or 2D text data. Ignored.
        y:
            Optional[Union[any, None]], default = None - the target for
            the data. Always ignored.


        Returns
        -------
        -
            self - the TextReplacer instance.


        """


        return self


    def fit(
        self,
        X: XContainer,
        y: Optional[Union[any, None]] = None
    ) -> Self:

        """
        No-op one-shot fit of the TextReplacer instance.


        Parameters
        ----------
        X:
            XContainer - 1D or 2D text data. Ignored.
        y:
            Optional[Union[any, None]], default = None - the target for
            the data. Always ignored.


        Returns
        -------
        -
            self - the TextReplacer instance.


        """


        return self.partial_fit(X, y)


    def transform(
        self,
        X:XContainer,
        copy:Optional[bool] = False
    ) -> XWipContainer:

        """
        Search the data for matches against the search criteria and make
        the specified replacements.


        Parameters
        ----------
        X:
            XContainer - 1D or 2D text data whose strings will be
            searched and may have substrings replaced.
        copy:
            Optional[bool], default=False - whether to make the
            replacements directly on the given X or on a deepcopy of X.


        Returns
        -------
        -
            XWipContainer: the data with replacements made.


        """


        check_is_fitted(self)

        _validation(
            X,
            self.replace,
            self.case_sensitive,
            self.flags
        )

        if copy:
            _X = copy_X(X)
        else:
            _X = X

        _X: XWipContainer = _map_X_to_list(_X)

        _rr: WipReplaceType = _special_param_conditioner(
            self.replace,
            self.case_sensitive,
            self.flags,
            _n_rows = len(_X)
        )

        if all(map(isinstance, _X, (str for _ in _X))):

            _X = _regexp_1D_core(_X, _rr)

        else:

            for _row_idx in range(len(_X)):

                _X[_row_idx] = _regexp_1D_core(
                    _X[_row_idx],
                    _rr[_row_idx] if isinstance(_rr, list) else _rr
                )

        del _rr

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
            XContainer - 1D or 2D text data. Ignored.
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









