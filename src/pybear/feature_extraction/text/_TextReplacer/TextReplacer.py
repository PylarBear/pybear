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

import re

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
    Search 1D vectors or (possibly ragged) 2D arrays of text data for
    character substrings and make one-to-one replacements. TextReplacer
    (TR) can search for patterns to replace by literal strings or regex
    patterns.

    So why not just use str.replace or re.sub? TextReplacer (TR) provides
    a few conveniences beyond the python built-ins. First, you can
    quickly apply multiple replacement criteria over the entire text
    body in one step. Second, you won't need to put in multiple search
    patterns to manage case-sensitivity. TR has a :param: `case-sensitive`
    parameter that allows you to globally toggle this behavior. Third,
    should you need granular control of replacements at the individual
    row level, TR allows customized search and replacement criteria for
    each row in your data. Finally, TR is a full-blown scikit-style
    transformer, and can be incorporated into larger workflows, like
    pipelines.

    For those who don't know how to write regular expressions or don't
    want to spend the time fine-tuning the patterns, TR's exact string
    matching provides quick access to search and replace functionality
    with multiple criteria. For those who do know regex, TR also accepts
    regular expressions via re.compile objects.

    TR accepts find/replace pairs in tuples to the :param: `replace`
    parameter. In the first position of the tuple, specify the substring
    pattern to be searched for in your text. Provide a literal string or
    re.compile object containing your regex pattern intended to match
    substrings. DO NOT PASS A REGEX PATTERN AS A LITERAL STRING. YOU
    WILL NOT GET THE CORRECT RESULT. ALWAYS PASS REGEX PATTERNS IN A
    re.compile OBJECT. DO NOT ESCAPE LITERAL STRINGS, TextReplacer WILL
    DO THAT FOR YOU. If you don't know what any of that means, then you
    don't need to worry about it.

    In the second position of the tuple, specify what to substitute
    in when a match against the corresponding pattern is found. The
    replacement value for your search pattern can be specified as a
    literal string or a callable that accepts the substring in your text
    that matched the pattern, does some operation on it, and returns a
    single string. An example regex replacement criteria might be
    (re.compile('[a-m]'), your_callable()).

    You can pass multiple find/replace tuples to :param: `replace`
    to quickly execute multiple search criteria on your data. Pass
    multiple find/replace tuples described above in one enveloping
    tuple. An example might be ((',', ''), ('.', ''), (';' '')). TR
    works from left to right through the tuple when searching and
    replacing. Knowing this, the replacements can be gamed so that
    later searches are dependent on earlier replacements.

    To make fine-grained replacements on the individual rows of your
    data, you can pass a python list of the above-described tuples and
    tuples-of-tuples. The length of the list must match the length of
    the data, for both 1D and 2D datasets. To turn off search/replace for
    particular rows of the data, put None in those indices in the list.
    So the list you pass to :param: `replace` can contain find/replace
    tuples, tuples of find/replace tuples, and None.

    TR searches always default to case-sensitive, but can be made to
    be case-insensitive. You can globally set this behavior via
    the :param: `case_sensitive` parameter. For those of you that know
    regex, you can also put flags in the re.compile objects passed
    to :param: `replace`, or flags can be set globally via :param: flags.
    Case-sensitivity is generally controlled by :param: `case_sensitive`
    but IGNORECASE flags passed via re.compile objects or :param: `flags`
    will always overrule `case_sensitive`. :param: `case_sensitive` also
    accepts lists so that you can control this behavior down to the
    individual row. When passed as a list, the number of entries
    in the list must equal the number of rows in the data. The list can
    contain True, False, and/or None. When None, the default of True is
    applied.

    If you need to use flags, they can be passed directly to a re.compile
    object in the search criteria. For example, a search criteria might
    be (re.compile('a', re.I), ''). In this case, the re.I flag will make
    that specific search case agnostic. re flags can be passed globally
    to the :param: `flags` parameter. Any flags passed globally will be
    joined with any flags passed to the individual compile objects by
    bit-wise OR. You can also exercise fine-grained control on certain
    rows of data for the `flags` parameter. When passed as a list, the
    number of entries in the list must equal the number of rows in the
    data. The list can contain re flags (integers) or None to not apply
    any (new) flags to that row. Even if None is passed to a particular
    index of the list, any flags passed to re.compile objects would
    still take effect.

    TR does not have a 'count' parameter as you would see with re.sub
    and str.replace. When replacement is not disabled for a certain row,
    TR always makes the specified substitution for everything that
    matches your pattern. In a way, TR has a more basic implementation
    of this functionality through its all-or-None behavior. You can pass
    a list to :param: `replace` and set the value for a particular row
    index of the data to None, in which case zero replacements will be
    made for that row. Otherwise, all replacements will be made on that
    row of data.

    TR can be instantiated with the default parameters, but this will
    result in a no-op. To actually make replacements, you must pass at
    least 1 find/replace pair to :param: `replace`.

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
    numpy arrays, pandas dataframes, and polars dataframes. When passed
    a 1D list-like, a python list of the same size is returned. When
    passed a possibly ragged 2D array-like, an identically-shaped list
    of python lists is returned.


    Parameters
    ----------
    replace:
        ReplaceType, default=None - the literal string pattern(s) or
        regex pattern(s) to search for and their replacement value(s).
    case_sensitive:
        Optional[CaseSensitiveType] - global setting for case-sensitivity.
        If True (the default) then all searches are case-sensitive. If
        False, TR will look for matches regardless of case. This setting
        is overriden when IGNORECASE flags are passed in re.compile
        objects or to :param: `flags`.
    flags:
        Optional[FlagsType] - the flags values(s) for the substring
        searches. Internally, TR does all its searching for substrings
        with re.sub, therefore flags can be passed whether you are
        searching for literal strings or regex patterns. If you do not
        know regular expressions, then you do not need to worry about
        this parameter. If None, the default flags for re.sub()
        are used globally. If a single flags object, that is applied
        globally. If passed as a list, the number of entries must match
        the number of rows in the data. Flags objects and Nones in the
        list follow the same rules stated above, but at the row level.
        If IGNORECASE is passed here as a global setting or in a list
        it overrides the :param: `case_sensitive` 'True' setting.


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
    re.sub


    Examples
    --------
    >>> from pybear.feature_extraction.text import TextReplacer as TR
    >>> trfm = TR(replace=((',', ''),(re.compile('\.'), '')))
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









