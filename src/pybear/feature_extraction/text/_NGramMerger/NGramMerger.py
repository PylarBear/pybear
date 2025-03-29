# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Callable, Optional, Sequence
from typing_extensions import Self, Union
from ._type_aliases import (
    XContainer,
    XWipContainer
)

import re

import pandas as pd
import polars as pl

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



class NGramMerger(
    FitTransformMixin,
    GetParamsMixin,
    ReprMixin,
    SetParamsMixin
):

    """
    Join specified adjacent words into an N-gram unit, to be handled as
    a single "word".

    Sometimes in text analytics it makes sense to work with a block of
    words as a single unit rather than as stand-alone words. Perhaps you
    have a project where you need to analyze the frequency of certain
    occupations or terms across hundreds of thousands of resumes. After
    starting your analysis, you realize that terms like 'process' and
    'engineer' or 'executive' and 'assistant' almost always seem to be
    with each other. So you decide that it might be meaningful to conduct
    your analysis as if those terms were a single unit.

    That's where NGramMerger comes in. NGramMerger (NGM) is a tool that
    will find blocks of words that you specify and join them by a
    separator of your choosing to create a single contiguous string.

    NGM works from top-to-bottom and left-to-right across the data,
    using a forward-greedy approach to merging n-grams. For example, if
    you passed an n-gram pattern of ['BACON', 'BACON'], and in the text
    body there is a line that contains ...'BACON', 'BACON', 'BACON', ...
    NGM will apply the n-gram as 'BACON_BACON', 'BACON'. This aspect of
    NGM's operation cannot be changed. When using wrapped searches (read
    on for more information about wrapping), the same forward-greedy
    technique is applied. Consider, for example, a case where a pattern
    match exists at the end of one line and into the next line, but in
    the next line there is an overlapping match. NGM will apply the
    wrapped match first because it is first in the working order, and
    consume the words out of the overlap, destroying the second matching
    pattern.

    NGM handles all pattern matches as mutually exclusive. Overlapping
    match patterns are not acknowledged. When NGM finds a pattern match,
    it will immediately jump to the next word AFTER the pattern, not
    the next word within the pattern.

    N-gram patterns can be built with literal strings, regex patterns,
    or re.compile objects. NGM always looks for full matches against
    tokens, it does not do partial matches. Pass literal strings or regex
    patterns that are intended to match entire words. NGM searches are
    always case-sensitive, unless you use flags in a re.compile object
    to tell it otherwise. String literal searches will always be
    case-sensitive.

    When you pass n-grams via :param: `ngrams` parameter, NGM does not
    necessarily run them in the given order. To prevent conflicts,
    NGM runs the n-gram patterns in descending order of length, that is,
    the longest n-gram is run first and the shortest n-gram is run last.
    For n-grams that are the same length, NGM runs them in the order
    that they were entered in :param: `ngrams`. If you would like to
    impose another n-gram run order hierarchy, you can manipulate the
    order in which NGM sees the n-grams by setting the n-grams piecemeal
    via :meth: `set_params`. Instantiate with your preferred n-grams,
    pass the data to :meth: `transform`, and keep the processed data in
    memory. Then use :meth: `set_params` to set the lesser-preferred
    n-grams and pass the processed data to :meth: `transform` again.

    NGM affords you some control over how the n-grams are merged in the
    text body. There are two parameters that control this, :param: `sep`
    and :param: `ngcallable`. :param: `ngcallable` allows you to pass a
    function that takes a variable-length list of strings and returns a
    single string. :param: `sep` will simply concatenate the words in
    the matching pattern with the separator that you choose. If you pass
    neither, NGM will default to concatenating the words in the matching
    pattern with a '_' separator. In short, NGM merges words that match
    ngram patterns using the following hierarchy:
    given callable > given separator > default separator

    NGM is able to wrap across the beginnings and ends of line, if you
    desire. This can be toggled with :param: `wrap`. If you do not want
    NGM to look for and join n-grams across the end of one line into
    the beginning of another, set this parameter to False (the default).
    When True, NGM will look for matches as if there is no break between
    the two lines. When allowed and an n-gram match is found across 2
    lines, the joined n-gram is put into the line where the match began.
    For example, if an n-gram match is found starting at the end of line
    724 and ends in the beginning of line 725, the joined n-gram will go
    at the end of line 724 and the words in line 725 will be removed.
    NGM only looks for wrapped n-grams across 2 lines, no further.
    Consider the case where you have text that is one word per line,
    and you are looking for a pattern like ['ONE', 'TWO', 'THREE']. NGM
    will not find a match for this across 3 lines. The way to match this
    n-gram would be 1) put all your tokens on one line, or 2) make 2
    passes. On, the first pass look for the n-gram ['ONE', 'TWO'], then
    on the second pass look for the n-gram ['ONE_TWO', 'THREE'].

    NGM should only be used on highly processed data. NGM should not be
    the first (or even near the first) step in a complex text wrangling
    workflow. This should be one of the last steps. An example of a
    text wrangling workflow could be: TextReplacer > TextSplitter >
    TextNormalizer > TextRemover > TextLookup > StopRemover > NGramMerger.

    NGM requires (possibly ragged) 2D data formats. The data should be
    processed at least to the point that you are able to split your data
    into tokens. (If you have 1D data and know what your separators are
    as either string literal or regex patterns, use pybear TextSplitter
    to convert your data to 2D before using NGM.) Accepted 2D objects
    include python list/tuple of lists/tuples, numpy arrays, pandas
    dataframes, and polars dataframes. Results are always returned as a
    python list of lists of strings.

    NGM is a full-fledged scikit-style transformer. It has fully
    functional get_params, set_params, transform, and fit_transform
    methods. It also has partial_fit, fit, and score methods, which are
    no-ops. NGM technically does not need to be fit because it already
    knows everything it needs to do transformations from the parameters.
    These no-op methods are available to fulfill the scikit transformer
    API and make NGM suitable for incorporation into larger workflows,
    such as Pipelines and dask_ml wrappers.

    Because NGM doesn't need any information from :meth: `partial_fit`
    and :meth: `fit`, it is technically always in a 'fitted' state and
    ready to transform data. Checks for fittedness will always return
    True.

    NGM has 2 attributes which are only available after data has been
    passed to :meth: `transform`. :attr: `n_rows_` is the number of rows
    of text seen in the original data, which may not be equal to the
    number of rows in the outputted data. :attr: `row_support_` is a 1D
    boolean vector that indicates which rows were kept (True) and which
    rows were removed (False) from the data during :term: transform.
    The only way for an entry to become False (i.e. a row was removed)
    is if both :param: `wrap` and :param: `remove_empty_rows` are True
    and all the strings on one row are merged into an n-gram at the end
    of the line above it. :attr: `n_rows_` must equal the number of
    entries in :attr: `row_support_`.


    Parameters
    ----------
    ngrams:
        Sequence[Sequence[Union[str, re.Pattern]]] - A sequence of
        sequences, where each inner sequence holds a series of string
        literals and/or re.Pattern objects that specify an n-gram.
        Cannot be empty, and cannot have any n-gram patterns with less
        than 2 entries.
    ngcallable:
        Optional[Callable[[Sequence[str]], str]], default=None - a
        callable applied to word sequences that match an n-gram to
        produce a contiguous string sequence.
    sep:
        Optional[str], default='_' - the separator that joins words that
        match an n-gram.
    wrap:
        Optional[bool], default=False - whether to look for pattern
        matches across the end of the current line and beginning of the
        next line.
    remove_empty_rows:
        Optional[bool], default=False - whether to delete any empty rows
        that may occur during the merging process. A row could only
        become empty if :param: `wrap` is True.


    Attributes
    ----------
    n_rows_:
        int - the number of rows in the data passed to :meth: `transform`.
        This reflects the data that is passed, not the data that is
        returned, which may not necessarily have the same number of
        rows as the original data if :param: `remove_empty_rows`
        and :param: `wrap` and are both True. Also, it only reflects the
        last batch of data passed; it is not cumulative. This attribute
        is only exposed after data is passed to :meth: `transform`.
    row_support_:
        np.NDArray[bool] - a boolean 1D numpy vector of shape (n_rows_, )
        indicating which rows of the data were kept (True) or removed
        (False) during :term: transform. The only way an entry in
        this vector could become False (i.e. a row was removed) is
        if :param: `wrap` and :param: `remove_empty_rows` are both True
        and all strings on one line were merged into an n-gram on the
        line above it. This attribute is exposed after data is passed
        to :meth: `transform`.


    Notes
    -----
    Type Aliases

    PythonTypes:
        Sequence[Sequence[str]]

    NumpyTypes:
        npt.NDArray[str]

    PandasTypes:
        pd.DataFrame

    PolarsTypes:
        pl.DataFrame

    XContainer:
        Union[PythonTypes, NumpyTypes, PandasTypes, PolarsTypes]

    XWipContainer:
        list[list[str]]

    NGramsType:
        Sequence[Sequence[Union[str, re.Pattern]]]

    CallableType:
        Optional[Callable[[Sequence[str]], str]]

    SepType:
        Optional[str]

    WrapType:
        Optional[bool]

    RemoveEmptyRowsType:
        Optional[bool]


    Examples
    --------
    >>> from pybear.feature_extraction.text import NGramMerger as NGM
    >>> trfm = NGM(ngrams=[('NEW', 'YORK', 'CITY'), ('NEW', 'YORK')])
    >>> X = [
    ...   ['UNITED', 'NATIONS', 'HEADQUARTERS'],
    ...   ['405', 'EAST', '42ND', 'STREET'],
    ...   ['NEW', 'YORK', 'CITY', 'NEW', 'YORK', '10017', 'USA']
    ... ]
    >>> out = trfm.fit_transform(X)
    >>> for line in out:
    ...   print(line)
    ['UNITED', 'NATIONS', 'HEADQUARTERS']
    ['405', 'EAST', '42ND', 'STREET']
    ['NEW_YORK_CITY', 'NEW_YORK', '10017', 'USA']
    >>> # Change the separator to '@'
    >>> trfm.set_params(sep='@')
    NGramMerger(ngrams=[('NEW', 'YORK', 'CITY'), ('NEW', 'YORK')], sep='@')
    >>> out = trfm.fit_transform(X)
    >>> for line in out:
    ...   print(line)
    ['UNITED', 'NATIONS', 'HEADQUARTERS']
    ['405', 'EAST', '42ND', 'STREET']
    ['NEW@YORK@CITY', 'NEW@YORK', '10017', 'USA']


    """


    def __init__(
        self,
        *,
        ngrams: Sequence[Sequence[Union[str, re.Pattern]]],
        ngcallable: Optional[Callable[[Sequence[str]], str]]=None,
        sep: Optional[str]='_',
        wrap: Optional[bool]=False,
        remove_empty_rows: Optional[bool]=False
    ) -> None:

        """Initialize the NGramMerger instance."""

        self.ngrams = ngrams
        self.ngcallable = ngcallable
        self.sep = sep
        self.wrap = wrap
        self.remove_empty_rows = remove_empty_rows


    @property
    def n_rows_(self):
        """
        Get the 'n_rows_' attribute. The number of rows in the data
        passed to transform.
        """
        return self._n_rows


    @property
    def row_support_(self):
        """
        Get the row_support_ attribute. A boolean vector indicating
        which rows were kept in the data during the transform process.
        Only available if a transform has been performed, and only
        reflects the results of the last transform done.
        """
        return self._row_support


    def __pybear_is_fitted__(self):
        return True


    # def get_params
    # handled by GetParamsMixin


    # def set_params
    # handled by SetParamsMixin


    # def fit_transform
    # handled by FitTransformMixin


    def reset(self) -> Self:
        """No-op reset method."""
        return self


    def get_metadata_routing(self):
        raise NotImplementedError(
            f"'get_metadata_routing' is not implemented in NGramMerger"
        )


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
            XContainer - The data. Ignored.
        y:
            Optional[Union[any, None]], default=None - the target for
            the data. Always ignored.


        Return
        ------
        -
            self - the NGramMerger instance.


        """


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
            XContainer - The data. Ignored.
        y:
            Optional[Union[any, None]], default=None - the target for
            the data. Always ignored.


        Return
        ------
        -
            self - the NGramMerger instance.


        """

        self.reset()

        return self.partial_fit(X, y)


    def transform(
        self,
        X:XContainer,
        copy:Optional[bool] = False
    ) -> XWipContainer:

        """
        Merge N-grams in a (possibly ragged) 2D array-like of strings.


        Parameters
        ----------
        X:
            XContainer - The data.
        copy:
            Optional[bool], default=False - whether to directly operate
            on the passed X or on a deepcopy of X.


        Return
        ------
        -
            list[list[str]] - the data with all matching n-gram patterns
            replaced with contiguous strings.


        """

        check_is_fitted(self)

        _validation(
            X, self.ngrams, self.ngcallable, self.sep, self.wrap,
            self.remove_empty_rows
        )

        if copy:
            _X = copy_X(X)
        else:
            _X = X

        if isinstance(_X, pd.DataFrame):
            _X = list(map(list, _X.values))
        elif isinstance(_X, pl.DataFrame):
            _X = list(map(list, _X.rows()))
        else:
            _X = list(map(list, _X))

        self._n_rows: int = len(_X)

        _X, self._row_support = _transform(
            _X, self.ngrams, self.ngcallable, self.sep, self.wrap,
            self.remove_empty_rows
        )

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
            XContainer - The data. Ignored.
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





