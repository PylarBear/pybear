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




class TextJustifier(
    FitTransformMixin,
    GetParamsMixin,
    ReprMixin,
    SetParamsMixin
):

    """
    TJ will not autonomously hyphenate words.

    Say something about how if on the first 'token' of a line, and the number
    of characters to the next split location is greater than 'n_chars',
    TJ will allow that first token to extend beyond n_chars as much as it
    needs to. This is to handle the edge case of small 'n_chars', where
    'n_chars' might be lower then a typical 'token' length. This only
    time and place where this will happen; all other circumstances after
    the first token defined by the first instance of a 'sep' will wrap.

    add something about no validation of the value of sep.


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
        Optional[Union[str, set[str]]], default=' ' - the character
        string sequence(s) that indicate to TextJustifier where it is
        allowed to wrap a line. When passed as a set of strings,
        TextJustifier will consider any of those strings as a place where
        it can wrap a line; cannot be empty.
        TextJustifier processes all data in 1D form (as list of strings),
        with all data given as 2D converted to 1D. If a sep string is in
        the middle of a 'token', or some other sequence that would
        otherwise be expected to be contiguous, TJ will split on that
        spot if it determines to do so. It will wrap a new line
        immediately after the matching string indiscriminately.
    line_break:
        Optional[Union[str, set[str], None]], default=None - When passed
        as a single string, TextJustifier will start a new line
        immediately AFTER all occurrences of the character string
        sequence. When passed as a set of strings, TextJustifier will
        start a new line immediately after all occurrences of the
        character strings given; cannot be empty. If None, do not force
        any line breaks. If the there are no string sequences in the
        data that match the given strings, then there are no forced line
        breaks. If a line_break string is in the middle of a 'token', or
        some other sequence that would otherwise be expected to be
        contiguous, TJ will not preserve the whole token. It will start
        a new line immediately after the matching string indiscriminately.
    backfill_sep:
        Optional[str], default=' ' - Some of the lines in your text may
        not have any of the wrap separators or line breaks you have
        specified. When justifying text and there is a shortfall of
        characters in a line, TJ will look to the next line to backfill
        strings. In the case where the line being backfilled onto does
        not have a separator at the end of the string, this character
        string will separate the otherwise separator-less strings from
        the strings being backfilled onto them. If you do not want a
        separator in this case, pass an empty string to this parameter.
    join_2D:
        Optional[Union[str, Sequence[str]]], default=' ' - Ignored if
        the data is given as a 1D sequence. For 2D
        containers of (perhaps token) strings, the character string
        sequence(s) that are used to join the strings across rows. If a
        single string, that value is used to join for all rows. If a
        sequence of strings, then the number of strings in the sequence
        must match the number of rows in the data, and each entry in the
        sequence is applied to the corresponding entry in the data.


    Attributes
    ----------
    n_rows_:
        int - the number of rows of text seen during transform and the
        number of strings in the returned 1D python list.


    Notes
    -----
    Type Aliases

    PythonTypes:
        Union[Sequence[str], Sequence[Sequence[str]]]

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


    """

    def __init__(
        self,
        *,
        n_chars:Optional[numbers.Integral] = 79,
        sep:Optional[Union[str, set[str]]] = ' ',
        line_break:Optional[Union[str, set[str], None]] = None,
        backfill_sep:Optional[str] = ' ',
        join_2D:Optional[Union[str, Sequence[str]]] = ' '
    ) -> None:

        """Initialize the TextJustifier instance."""

        self.n_chars = n_chars
        self.sep = sep
        self.line_break = line_break
        self.backfill_sep = backfill_sep
        self.join_2D = join_2D



    def __pybear_is_fitted__(self):
        return True


    def get_metadata_routing(self):
        raise NotImplementedError(
            f"'get_metadata_routing' is not implemented in TextJustifier"
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
            Union[Sequence[str], Sequence[Sequence[str]]] - The data to
            justify. Ignored.
        y:
            Optional[Union[any, None]], default=None - the target for
            the data. Always ignored.


        Return
        ------
        -
            self - the TextJustifier instance.


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
            Union[Sequence[str], Sequence[Sequence[str]]] - The data to
            justify. Ignored.
        y:
            Optional[Union[any, None]], default=None - the target for
            the data. Always ignored.


        Return
        ------
        -
            self - the TextJustifier instance.


        """

        return self.partial_fit(X, y)


    def transform(
        self,
        X: XContainer,
        copy: Optional[bool] = True
    ) -> OutputContainer:

        """
        Justify the text in a 1D list-like of strings or a (possibly
        ragged) 2D array like of strings.


        Parameters
        ----------
        X:
            Union[Sequence[str], Sequence[Sequence[str]]] - The data to
            justify.
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
            X, self.n_chars, self.sep, self.line_break,
            self.backfill_sep, self.join_2D
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
            _X, self.n_chars, self.sep,
            self.line_break, self.backfill_sep
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
            Union[Sequence[str], Sequence[Sequence[str]]] - The data to
            justify. Ignored.
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





