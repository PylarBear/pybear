# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Callable, Optional
from typing_extensions import Self, Union
from ._type_aliases import (
    XContainer,
    XWipContainer,
    RowSupportType
)

import numbers
import re

import pandas as pd
import polars as pl

from ._transform._transform import _transform
from ._validation._validation import _validation

from .._Lexicon.Lexicon import Lexicon

from ....base import (
    FitTransformMixin,
    GetParamsMixin,
    ReprMixin,
    SetParamsMixin,
    check_is_fitted
)

from ....base._copy_X import copy_X



class StopRemover(
    FitTransformMixin,
    GetParamsMixin,
    ReprMixin,
    SetParamsMixin
):

    """
    Remove stop words from text data.

    StopRemover (SR) uses the stop words defined in the pybear Lexicon
    'stop_words_' attribute to locate and remove stop words from a 2D
    array-like body of text data.

    pybear wants to deliver a robust and predictable output for your
    inputs, and recommends that SR should only be used on highly
    processed data. SR should not be the first (or even near the first)
    step in a complex text wrangling workflow. This should be one of the
    last steps. An example of the last steps of a workflow using pybear
    text wrangling modules could be: ... > TextLookup > StopRemover >
    NGramMerger > TextJustifier.

    To this end, SR only accepts tokenized text in 2D array-like format.
    Trying to manage the contingencies of replacing stop words vis-a-vis
    individual user preferences concerning adjoining characters and the
    impact on white space in long text strings as would be in 1D format
    is intractable. Therefore, pybear pushes back to the user to require
    that the data be processed at least to the point where you know what
    your separators are and you are able to split your data into tokens.
    (If you have 1D data and know what your separators are as either
    string literal or regex patterns, use pybear TextSplitter to convert
    your data to 2D before using SR.) Accepted 2D objects include python
    list/tuple of lists/tuples, numpy arrays, pandas dataframes, and
    polars dataframes. Results are always returned as a python list of
    lists of strings.

    The default text comparer in SR does a case-insensitive, exact
    character-to-character match of each token in the text body against
    the stop words, and removes a word from the text when there is a
    match. If you want to override the default SR case-insensitive
    behavior, pass a new callable to the 'match_callable' parameter. The
    callable can take anything that you can put into a callable, as long
    as the signature is [str, str] and returns a boolean. If you would
    like to do your stop word matching with regular expressions then
    by all means put that in your callable.

    Optionally, you can instruct SR to remove any empty rows that may be
    left after the stop word removal process. After transform, SR
    exposes a 'row_support_' attribute which is a boolean vector that
    shows which rows in the data were kept (True). The only way an entry
    in this vector could become False is if the 'remove_empty_rows'
    parameter is True and a row became empty during the stop word
    removal process. row_support_ only reflects the last dataset passed
    to transform.

    SR is a full-fledged scikit-style transformer. It has fully
    functional get_params, set_params, transform, and fit_transform
    methods. It also has partial_fit, fit, and score methods, which are
    no-ops. SR technically does not need to be fit because it already
    knows everything it needs to do transformations from the parameters
    and the stop words in the pybear Lexicon. These no-op methods are
    available to fulfill the scikit transformer API and make SR suitable
    for incorporation into larger workflows, such as Pipelines and
    dask_ml wrappers.

    Because SR doesn't need any information from partial_fit and fit, it
    is technically always in a 'fitted' state and ready to transform
    data. Checks for fittedness will always return True.

    SR has an n_rows_ attribute which is only available after data has
    been passed to :method: transform. n_rows_ is the number of rows
    of text seen in the original data, and must match the number of
    entries in row_support_.


    Parameters
    ----------
    match_callable:
        Optional[Union[Callable[[str, str], bool], None]], default=None -
        None to use the default StopRemover matching criteria, or a
        custom callable that defines what constitutes matches of words
        in the text against the stop words. In pre-run validation, SR
        only checks that 'match_callable' is None or a callable, no
        validation is done on the callable. It is a heavy burden to
        validate the user-defined callable at every call over a search
        of the entire text body for every stop word, so SR does not
        validate any of it. If the user-defined callable is ill-formed,
        SR could break in unpredictable ways, or, perhaps worse, SR may
        not break and successfully complete the search operation and
        yield nonsensical results. It is up to the user to validate the
        accuracy of their callable and ensure that the output is a
        boolean.
    remove_empty_rows:
        Optional[bool], default=True - whether to remove any rows that
        are left empty by the stop word removal process.
    exempt:
        Optional[Union[list[str], None]], default=None - stop words that
        are exempted from the search. Text that matches these words will
        not be removed. Ensure that the capitalization of the word(s)
        that you enter exactly matches that of the word(s) in the
        Lexicon. Always enter words in majuscule if working with the
        default pybear Lexicon.
    supplemental:
        Optional[Union[list[str], None]], default=None - words to be
        removed in addition to the stop words. If you intend to do a
        case-sensitive search then the capitalization of these words
        matters.
    n_jobs:
        Optional[Union[numbers.Integral, None]], default = -1 - the
        number of cores/threads to use when parallelizing the search for
        stop words in the rows of X. The default is to use processes but
        can be set by running StopRemover under a joblib parallel_config
        context manager. None uses the default number of cores/threads.
        -1 uses all available cores/threads.


    Attributes
    ----------
    n_rows_:
        int - the number of rows in the data passed to transform.
    row_support_:
        NDArray[bool] of shape (n_original_rows, ) - a 1D boolean numpy
        vector indicating which rows have been kept (True) after the
        stop word removal process. Entries in this vector could only
        become False if :param: 'remove_empty_rows' is True and one or
        more rows became empty during the transform process. row_support_
        only reflects the last dataset passed to transform.


    Notes
    -----
    Type Aliases

    PythonTypes:
        Sequence[Sequence[str]]

    NumpyTypes:
        npt.NDArray

    PandasTypes:
        pd.DataFrame

    PolarsTypes:
        pl.DataFrame

    XContainer:
        Union[PythonTypes, NumpyTypes, PandasTypes, PolarsTypes]

    XWipContainer:
        list[list[str]]

    RowSupportType:
        npt.NDArray[bool]


    Examples
    --------
    >>> from pybear.feature_extraction.text import StopRemover as SR
    >>> trfm = SR(remove_empty_rows=True, n_jobs=1)
    >>> X = [
    ...     ['but', 'I', 'like', 'to', 'be', 'here'],
    ...     ['oh', 'I', 'like', 'it', 'a', 'lot'],
    ...     ['said', 'the', 'cat', 'in', 'the', 'that'],
    ...     ['to', 'the', 'fish', 'in', 'the', 'pot']
    ... ]
    >>> trfm.transform(X)
    [['oh', 'lot'], ['cat'], ['fish', 'pot']]


    """

    def __init__(
        self,
        match_callable: Optional[Union[Callable[[str, str], bool], None]] = None,
        remove_empty_rows: Optional[bool] = True,
        exempt: Optional[Union[list[str], None]] = None,
        supplemental: Optional[Union[list[str], None]] = None,
        n_jobs: Optional[Union[numbers.Integral, None]] = -1
    ) -> None:
        """Initialize the StopRemover instance."""

        self.match_callable = match_callable
        self.remove_empty_rows = remove_empty_rows
        self.exempt = exempt
        self.supplemental = supplemental
        self.n_jobs = n_jobs

        self._stop_words = Lexicon().stop_words_
        self._stop_words = set(self._stop_words) - set(self.exempt or [])
        self._stop_words = set(self._stop_words).union(self.supplemental or [])
        self._stop_words = list(self._stop_words)


    def __pybear_is_fitted__(self):
        return True


    def get_metadata_routing(self):
        raise NotImplementedError(
            f"'get_metadata_routing' is not implemented in StopRemover"
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
        No-op batch-wise fit method.


        Parameters
        ----------
        X:
            XContainer - the (possibly ragged) 2D container of text from
            which to remove stop words. Ignored.
        y:
            Optional[Union[any, None]], default=None - the target for
            the data. Always ignored.


        Return
        ------
        -
            self: the StopRemover instance.


        """

        return self


    def fit(
        self,
        X: XContainer,
        y: Optional[Union[any, None]] = None
    ) -> Self:

        """
        No-op one-shot fit method.


        Parameters
        ----------
        X:
            XContainer - the (possibly ragged) 2D container of text from
            which to remove stop words. Ignored.
        y:
            Optional[Union[any, None]], default=None - the target for
            the data. Always ignored.


        Return
        ------
        -
            self: the StopRemover instance.


        """

        return self.partial_fit(X, y)


    def score(
        self,
        X: XContainer,
        y: Optional[Union[any, None]] = None
    ) -> Self:

        """
        No-op score method. Needs to be here for dask_ml wrappers.


        Parameters
        ----------
        X:
            XContainer - the (possibly ragged) 2D container of text from
            which to remove stop words. Ignored.
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


    @staticmethod
    def _default_callable(_str1:str, _str2:str) -> bool:
        """
        The default function for determining equality between a word
        in the text body and a stop word. This can be supplanted by
        passing a new callable with signature [str, str] to the
        'match_callable' parameter.
        """

        __ = re.fullmatch(re.compile(_str1, re.I), _str2)

        return (__ is not None and __.span() != (0, 0))


    def transform(
        self,
        X: XContainer,
        copy:Optional[bool] = True
    ) -> list[list[str]]:

        """
        Scan X and remove any stop words as defined in the pybear
        Lexicon 'stop_words_' attribute. Optionally removes any empty
        rows left by the stop word removal process. Exposes the n_rows_
        and row_support_ attributes. row_support_ is a boolean numpy
        vector that indicates which rows in the original X were kept
        during transform (True); entries could only become False if
        parameter 'remove_empty_rows' is True and at least one row
        became empty during the stop word removal process. row_support_
        only reflects the last dataset passed to transform.


        Parameters
        ----------
        X:
            XContainer - the (possibly ragged) 2D container of text from
            which to remove stop words.
        copy:
            Optional[bool], default=True - whether to remove stop words
            directly from the passed X or a deepcopy of X.


        Return
        ------
        -
            list[list[str]] - the data with stop words removed.


        """


        _validation(
            X,
            self.match_callable,
            self.remove_empty_rows,
            self.exempt,
            self.supplemental,
            self.n_jobs
        )

        if copy:
            _X = copy_X(X)
        else:
            _X = X

        # convert X to list-of-lists -- -- -- -- -- -- -- -- -- -- --
        # we know from validation it is legit 2D
        if isinstance(_X, pd.DataFrame):
            _X = list(map(list, _X.values))
        elif isinstance(_X, pl.DataFrame):
            _X = list(map(list, _X.rows()))
        else:
            _X = list(map(list, _X))

        _X: XWipContainer

        self.n_rows_ = len(_X)
        # END convert X to list-of-lists -- -- -- -- -- -- -- -- -- --

        _X, self.row_support_ = _transform(
            _X,
            self.match_callable or self._default_callable,
            self._stop_words,
            self.remove_empty_rows,
            self.n_jobs
        )


        return _X





