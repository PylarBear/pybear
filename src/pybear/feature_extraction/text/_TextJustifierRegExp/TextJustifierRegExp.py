# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Optional, Sequence
from typing_extensions import Self, Union
from ._type_aliases import (
    XContainer,
    OutputContainer
)

import numbers
import re

import pandas as pd
import polars as pl

from ._validation._validation import _validation
from ._transform._transform import _transform

from .._TextJoiner.TextJoiner import TextJoiner

from ....base import (
    FitTransformMixin,
    GetParamsMixin,
    ReprMixin,
    SetParamsMixin,
    check_is_fitted
)

from ....base._copy_X import copy_X



class TextJustifierRegExp(
    FitTransformMixin,
    GetParamsMixin,
    ReprMixin,
    SetParamsMixin
):

    """
    TextJustifierRegExp (TJRE) justifies text as closely as possible to
    the number of characters per line given by the user.

    This is not designed for making final drafts of highly formatted
    business letters. This is a tool designed to turn highly ragged
    text into block form that is more easily ingested and manipulated by
    humans and machines. Consider lines read in from text files or
    scraped from the internet. Many times there is large disparity in
    the number of characters per line, some lines may have a few
    characters, and other lines may have thousands of characters (or
    more.) This tool will square-up the text for you.

    The cleaner your data is, the more powerful this tool is, and the
    more predicable are the results. TJRE in no way is designed to do
    any cleaning. See the other pybear text wrangling modules for that.
    While TJRE will handle any text passed to it and blindly apply the
    instructions given to it, results are better when this is used
    toward the end of a text processing workflow. For best results,
    pybear recommends that removal of junk characters (pybear
    TextReplacer), empty strings (pybear TextRemover), and extra spaces
    (pybear TextStripper) be done before using TJRE.

    There are 3 operative parameters for justifying text in this module,
    'n_chars', 'sep', and 'line_break'. 'n_chars' is the target number
    of characters per line. The minimum allowed value is 1, and there is
    no maximum value. 'sep' is the regexp pattern that tells TJRE where
    it is allowed to wrap text. It does not mean that TJRE WILL wrap
    that particular text, but that it can if it needs to when near the
    n_chars limit on a line. The wrap occurs AFTER the sep pattern. A
    common 'sep' is a single space. 'line_break' is the regexp pattern
    that tells TJRE where it MUST wrap text. When TJRE finds a line_break
    pattern it will force a new line. The break occurs AFTER the
    line_break pattern. A typical 'line_break' might be a period.

    This tool is more advanced than pybear TextJustifier because it
    takes regular expressions. Whereas TextJustifier has validation and
    protections in place that prevent conflicts that could lead to
    undesired results, these safeguards are not in place here. In TJRE,
    a conflict exists when both the sep pattern and the line_break
    pattern identify the same location in text as the first character of
    a match. In that case, TJRE applies sep. Conflicts are impossible to
    predict when using regexp unless you know the text it is applied to
    beforehand. It is up to the user to assess the pitfalls and the
    likelihood of error when using regexps on their data. The user
    should inspect their results to ensure the desired outcome.

    But as simple as the tool is in concept, there are some nuances.
    Here is a non-exhaustive list of some of the quirks that may help
    the user understand some edge cases and explain why TJRE returns the
    things that it does.
    TJRE will not autonomously hyphenate words.
    If a line has no separators or line-breaks in it, then TJRE does
    nothing with it. If a line is millions of characters long and there
    are no places to wrap, TJRE will return the line as given, regardless
    of what n_chars is.
    If the margin is set very low, perhaps lower than the length of
    words (tokens) that may normally be encountered, then those
    words/lines will extend beyond 'n_chars'. Cool trick: if you want an
    itemized list of all the tokens in your text, set 'n_chars' to 1.

    TJRE accepts 1D and 2D data formats. Accepted objects include python
    built-in lists, tuples, and sets, numpy arrays, pandas series and
    dataframes, and polars series and dataframes. Results are always
    returned as a 1D python list of strings. When you pass your data in
    a 2D format, TJRE uses pybear TextJoiner to convert it to a 1D list.
    See TextJoiner for more information.

    TJRE is a full-fledged scikit-style transformer. It has fully
    functional get_params, set_params, transform, and fit_transform
    methods. It also has partial_fit, fit, and score methods, which are
    no-ops. TJRE technically does not need to be fit because it already
    knows everything it needs to do transformations from the parameters.
    These no-op methods are available to fulfill the scikit transformer
    API and make TJRE suitable for incorporation into larger workflows,
    such as Pipelines and dask_ml wrappers.

    Because TJRE doesn't need any information from partial_fit and fit,
    it is technically always in a 'fitted' state and ready to transform
    data. Checks for fittedness will always return True.

    TJRE has one attribute, n_rows_, which is only available after data
    has been passed to :method: transform. n_rows_ is the number of rows
    of text seen in the original data.


    Parameters
    ----------
    X:
        XContainer - the text to be justified. 2D containers can be
        ragged. 2D containers are converted to 1D for processing and are
        returned as 1D.
    n_chars:
        Optional[numbers.Integral], default=79 - the target number of
        characters per line when justifying the given text. Minimum
        allowed value is 1; there is no maximum value. Under normal
        expected operation with reasonable margins, the outputted text
        will not exceed this number but can fall short. If margins are
        unusually small, the output can exceed the given margins (e.g.
        the margin is set lower than an individual word's length.)
    sep:
        Optional[Union[str, re.Pattern], default='\s' - the regexp pattern
        that indicates to TJRE where it is allowed to wrap a line. If a
        pattern match is in the middle of a sequence that might otherwise
        be expected to be contiguous, TJRE will wrap a new line AFTER the
        pattern indiscriminately if proximity to the n_chars limit
        dictates to do so. This parameter is only validated by TJRE to
        be an instance of str or re.Pattern. TJRE does not assess the
        validity of the expression itself. Any exceptions would be
        raised by re.search.
    sep_flags:
        Optional[Union[numbers.Integral, None]], default=None, the flags
        for the 'sep' pattern. This parameter is only validated by TJRE
        to be an instance of numbers.Integral or None. TJRE does not
        assess the validity of the value. Any exceptions would be raised
        by re.search.
    line_break:
        Optional[Union[str, re.Pattern, None]], default=None - the
        regexp pattern that tells TJRE where it must start a new line.
        A new line will be started immediately AFTER all occurrences of
        the pattern regardless of the number of characters in the line.
        If None, do not force any line breaks. If the there are
        no patterns in the data that match, then there are no forced
        line breaks. If a line_break pattern is in the middle of a
        sequence that might otherwise be expected to be contiguous, TJRE
        will force a new line after the line_break indiscriminately.
    line_break_flags:
        Optional[Union[numbers.Integral, None]], default=None, the flags
        for the 'line_break' pattern. This parameter is only validated
        by TJRE to be an instance of numbers.Integral or None. TJRE does
        not assess the validity of the value. Any exceptions would be
        raised by re.search.
    backfill_sep:
        Optional[str], default=' ' - Some lines in the text may not have
        any of the given wrap patterns or line break patterns at the end
        of the line. When justifying text and there is a shortfall of
        characters in a line, TJRE will look to the next line to backfill
        strings. In the case where the line being backfilled onto does
        not have a separator or line break at the end of the string,
        this character string will separate the otherwise separator-less
        strings from the strings being backfilled onto them. If you do
        not want a separator in this case, pass an empty string to this
        parameter.
    join_2D:
        Optional[Union[str, Sequence[str]]], default=' ' - Ignored if
        the data is given as a 1D sequence. For 2D containers of strings,
        the character string sequence(s) that are used to join the
        strings across rows. If a single string, that value is used to
        join for all rows. If a sequence of strings, then the number of
        strings in the sequence must match the number of rows in the
        data, and each entry in the sequence is applied to the
        corresponding entry in the data. See pybear TextJoiner for more
        information.


    Attributes
    ----------
    n_rows_:
        int - the number of rows of text seen during transform and the
        number of strings in the returned 1D python list.


    Notes
    -----
    Type Aliases

    PythonTypes:
        Union[Sequence[str], Sequence[Sequence[str]], set[str]]

    NumpyTypes:
        npt.NDArray

    PandasTypes:
        Union[pd.Series, pd.DataFrame]

    PolarsTypes:
        Union[pl.Series, pl.DataFrame]

    XContainer:
        Union[PythonTypes, NumpyTypes, PandasTypes, PolarsTypes]

    OutputContainer:
        list[str]


    Examples
    --------
    >>> from pybear.feature_extraction.text import TextJustifierRegExp as TJRE
    >>> trfm = TJRE(n_chars=70, sep='\s', backfill_sep=' ')
    >>> X = [
    ...     "Hello darkness, my old friend",
    ...     "I've come to talk with you again",
    ...     "Because a vision softly creeping",
    ...     "Left its seeds while I was sleeping",
    ...     "And the vision that was planted in my brain",
    ...     "Still remains",
    ...     "Within the sound of silence"
    ... ]
    >>> out = trfm.fit_transform(X)
    >>> out = list(map(str.strip, out))
    >>> for _ in out:
    ...     print(_)
    Hello darkness, my old friend I've come to talk with you again
    Because a vision softly creeping Left its seeds while I was sleeping
    And the vision that was planted in my brain Still remains Within the
    sound of silence


    """

    def __init__(
        self,
        *,
        n_chars:Optional[numbers.Integral] = 79,
        sep:Optional[Union[str, re.Pattern]] = '\s',
        sep_flags:Optional[Union[numbers.Integral, None]] = None,
        line_break:Optional[Union[str, re.Pattern, None]] = None,
        line_break_flags:Optional[Union[numbers.Integral, None]] = None,
        backfill_sep:Optional[str] = ' ',
        join_2D:Optional[Union[str, Sequence[str]]] = ' '
    ) -> None:

        """Initialize the TextJustifierRegExp instance."""

        self.n_chars = n_chars
        self.sep = sep
        self.sep_flags = sep_flags
        self.line_break = line_break
        self.line_break_flags = line_break_flags
        self.backfill_sep = backfill_sep
        self.join_2D = join_2D



    def __pybear_is_fitted__(self):
        return True


    def get_metadata_routing(self):
        raise NotImplementedError(
            f"'get_metadata_routing' is not implemented in TextJustifierRegExp"
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
        No-op batch-wise fit operation.


        Parameters
        ----------
        X:
            XContainer - The data to justify. Ignored.
        y:
            Optional[Union[any, None]], default=None - the target for
            the data. Always ignored.


        Return
        ------
        -
            self - the TextJustifierRegExp instance.


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
            XContainer - The data to justify. Ignored.
        y:
            Optional[Union[any, None]], default=None - the target for
            the data. Always ignored.


        Return
        ------
        -
            self - the TextJustifierRegExp instance.


        """

        return self.partial_fit(X, y)


    def transform(
        self,
        X: XContainer,
        copy: Optional[bool] = True
    ) -> OutputContainer:

        """
        Justify the text in a 1D list-like of strings or a (possibly
        ragged) 2D array-like of strings.


        Parameters
        ----------
        X:
            XContainer - The data to justify.
        copy:
            Optional[bool], default=True - whether to directly operate
            on the passed X or on a copy.


        Return
        ------
        -
            OutputContainer - the justified data returned as a 1D python
            list of strings.


        """

        check_is_fitted(self)

        _validation(
            X, self.n_chars, self.sep, self.sep_flags, self.line_break,
            self.line_break_flags, self.backfill_sep, self.join_2D
        )


        if copy:
            _X = copy_X(X)
        else:
            _X = X

        # we know from validation it is legit 1D or 2D, do the easy check
        if all(map(isinstance, _X, (str for _ in _X))):
            # then is 1D:
            _X = list(_X)
        else:
            # then could only be 2D, need to convert to 1D
            if isinstance(_X, pd.DataFrame):
                _X = list(map(list, _X.values))
            elif isinstance(_X, pl.DataFrame):
                _X = list(map(list, _X.rows()))
            else:
                _X = list(map(list, _X))

            _X = TextJoiner(sep=self.join_2D).fit_transform(_X)

        # _X must be 1D at this point
        self.n_rows_ = len(_X)

        _X = _transform(
            _X, self.n_chars, self.sep, self.sep_flags,
            self.line_break, self.line_break_flags, self.backfill_sep
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
            XContainer - The data to justify. Ignored.
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





