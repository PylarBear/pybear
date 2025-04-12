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
    ReturnDimType,
    ReplaceType,
    RemoveType,
    LexiconLookupType,
    NGramsType,
    GetStatisticsType
)

import numbers
import re
import warnings

import numpy as np

from ._validation._validation import _validation

from .._NGramMerger.NGramMerger import NGramMerger
from .._StopRemover.StopRemover import StopRemover
from .._TextJoiner.TextJoiner import TextJoiner
from .._TextJustifier.TextJustifier import TextJustifier
from .._TextLookup.TextLookupRealTime import TextLookupRealTime
from .._TextNormalizer.TextNormalizer import TextNormalizer
from .._TextRemover.TextRemover import TextRemover
from .._TextReplacer.TextReplacer import TextReplacer
from .._TextSplitter.TextSplitter import TextSplitter
from .._TextStatistics.TextStatistics import TextStatistics
from .._TextStripper.TextStripper import TextStripper

from ..__shared._transform._map_X_to_list import _map_X_to_list

from ....base._copy_X import copy_X as _copy_X
from ....base import (
    FitTransformMixin,
    GetParamsMixin,
    ReprMixin,
    SetParamsMixin,
    check_is_fitted
)



class AutoTextCleaner(
    FitTransformMixin,
    GetParamsMixin,
    ReprMixin,
    SetParamsMixin
):

    """
    This class is a convenience for streamlining basic everyday data
    cleaning needs. It is not meant to do highly specialized text
    cleaning operations (A transformer designed to do that using the
    same underling pybear sub-transformers might have 50 parameters; 15
    is enough.) If you cannot accomplish what you are trying to do with
    this module out of the box, then you will need to construct your own
    workflow piece by piece with the individual pybear text modules.

    AutoTextCleaner adds no new functionality beyond what is in the
    other pybear text transformers; it simply lines them up and runs
    them all at once with one call to :meth: `transform`. All the
    information about the inner workings of this module is available in
    the docs for the submodules.

    This methods does have parameters and attributes that are unique
    to it. The documentation here mostly highlights these unique
    characteristics and points the reader to other documentation for
    more information.

    AutoTextCleaner (ATC) combines the functionality of the pybear
    text transformers into one module. In one shot you can strip,
    normalize, replace, remove, and justifiy text. You can also
    cross-reference the text against the pybear Lexicon, handle unknown
    words, remove stops, and merge n-grams. All the while, ATC is
    capable of compiling statistics about the incoming and outgoing
    text.

    IMPORTANT: if you want to use the :param: `lexicon_lookup` parameter
    and check your text against the pybear Lexicon, remember that the
    Lexicon is majuscule and has no non-alpha characters. You MUST
    set :param: `normalize` to True to get meaningful results, or you
    risk losing content that is not the correct case.

    ATC is a full-fledged scikit-style transformer. It has fully
    functional get_params, set_params, transform, and fit_transform
    methods. It also has no-op partial_fit and fit methods to allow
    for integration into larger workflows, like scikit pipelines.
    Technically ATC does not need to be fit and is always in a fitted
    state (any 'is_fitted' checks of an instance will always return
    True) because ATC knows everything it needs to know to transform
    data from the parameters. It also has a no-op :meth: `score` method
    to allow dask_ml wrappers.

    ATC accepts 1D list-like and (possibly ragged) 2D array-likes of
    strings. Accepted 1D containers include python lists, tuples, and
    sets, numpy vectors, pandas series, and polars series. Accepted 2D
    containers include embedded python sequences, numpy arrays, pandas
    dataframes, and polars dataframes. When passed a 1D list-like,
    returns a python list of strings. When passed a 2D array-like,
    returns a python list of python lists of strings. If you pass your
    data as a dataframe with feature names, the feature names are not
    preserved.


    Parameters
    ----------
    # pizza
    universal_sep:
        Optional[str] -
    case_sensitive:
        Optional[bool] -
    global_flags:
        Optional[Union[numbers.Integral, None]] -
    remove_empty_rows:
        Optional[bool] -
    join_2D:
        Optional[str] -
    return_dim:
        Optional[ReturnDimType] -
    strip:
        Optional[bool] -
    replace:
        Optional[ReplaceType] -
    remove:
        Optional[RemoveType] -
    normalize:
        Optional[Union[bool, None]] -
    lexicon_lookup:
        Optional[LexiconLookupType] -
    remove_stops:
        Optional[bool] -
    ngram_merge:
        Optional[NGramsType] -
    justify:
        Optional[Union[numbers.Integral, None]] -
    get_statistics:
        Optional[Union[None, GetStatisticsType]] -


    Attributes
    ----------
    n_rows_:
        int - Get the `n_rows_` attribute. The total number of rows in
        data passed to :meth: `transform` between resets. This may
        not be the number of rows in the outputted data. Unlike most
        other pybear text transformers that expose an `n_rows_`
        attribute that is not cumulative, this particular attribute
        is cumulative across multiple calls to :meth: `transform`.
        The reason for the different behavior is that the cumulative
        behavior here aligns this attribute with the behavior
        of :attr: `before_statistics_` and :attr: `after_statistics_`,
        which compile statistics cumulatively across multiple calls
        to :meth: `transform`. This number is reset when the
        AutoTextCleaner instance is reset by calls to :meth: `fit`.
    row_support_:
        npt.NDArray[bool] - Get the `row_support_` attribute. A 1D
        boolean numpy vector indicating which rows of the data, if
        any, were removed during the cleaning process. The length
        must equal the number of rows in the data originally passed
        to :meth: `transform`. A row that was removed is indicated by
        a False in the corresponding position in the vector, and a
        row that remains is indicated by True. This attribute only
        reflects the last batch of data passed to :meth: `transform`;
        it is not cumulative.
    before_statistics_:
        instance TextStatistics - Get the `before_statistics_` attribute.
        If the 'before' key of the :param: `get_statistics` parameter
        has a value of True or False, then statistics about the raw
        data were compiled in a TextStatistics instance before the
        transformation. This exposes that TextStatistics class (which
        is different from the :attr: `after_statistics_` TextStatistics
        class.) The exposed class has attributes that contain information
        about the raw data. See the documentation for TextStatistics to
        learn about what attributes are exposed. The statistics in this
        attribute are reset when the AutoTextCleaner instance is reset
        by calls to :meth: `fit`.
    after_statistics_:
        instance TextStatistics - Get the `after_statistics_` attribute.
        If the 'after' key of the :param: `get_statistics` parameter has
        a value of True or False, then statistics about the transformed
        data were compiled in a TextStatistics instance after the
        transformation. This exposes that TextStatistics class (which
        is different from the :attr: `before_statistics_` TextStatistics
        class.) The exposed class has attributes that contain information
        about the transformed data. See the documentation for
        TextStatistics to learn about what attributes are exposed. The
        statistics in this attribute are reset when the AutoTextCleaner
        instance is reset by calls to :meth: `fit`.
    lexicon_lookup_:
        instance TextLookupRealTime - Get the `lexicon_lookup_`
        attribute. If :param: `lexicon_lookup` has a non-None value,
        then information about the text-lookup process is stored in a
        TextLookupRealTime (TLRT) instance within ATC. This attribute
        exposes that TLRT class, which has attributes that contain
        information about the handling of words not in the pybear
        Lexicon. If you ran `lexicon_lookup` in 'manual' mode, you
        may have put a lot of effort into handing the unknown words
        and you want access to the information. You may have instructed
        TLRT to hold words that you want to add to the Lexicon so that
        you can access them later and put them in the Lexicon. See the
        documentation for TextLookupRealTime to learn about what
        attributes are exposed. The information in TLRT is reset when
        AutoTextCleaner is reset by calls to :meth: `fit`.


    Notes
    -----
    Type Aliases

    PythonTypes:
        Union[Sequence[str], set[str], Sequence[Sequence[str]]]

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
    
    ReturnDimType:
        Union[None, Literal['1D', '2D']]
    
    FindType:
        Union[str, re.Pattern[str]]
    
    SubstituteType:
        Union[str, Callable[[str], str]]
    
    PairType:
        tuple[FindType, SubstituteType]
    
    ReplaceType:
        Union[None, PairType, tuple[PairType, ...]]
    
    RemoveType:
        Union[None, FindType, tuple[FindType, ...]]
    
    LexiconLookupType:
        Union[None, Literal['auto_add', 'auto_delete', 'manual']]
    
    NGramsType:
        Union[Sequence[Sequence[FindType]], None]
    
    class GetStatisticsType(TypedDict):
        before: Required[Union[None, bool]]
        after: Required[Union[None, bool]]


    Examples
    --------
    >>> from pybear.feature_extraction.text import AutoTextCleaner as ATC
    >>> Trfm = ATC(case_sensitive=False, strip=True, remove_empty_rows=True,
    ...     replace=(re.compile('[^a-z]'), ''), remove='', normalize=True,
    ...     universal_sep=' ', get_statistics={'before': None, 'after':False},
    ...     lexicon_lookup='auto_delete', justify=30)
    >>> X = [
    ...       r' /033[91](tHis)i@s# S@o#/033[0m$e$tERR#I>B<Le te.X@t###dAtA. ',
    ...       r'@c.lE1123,AnIt up R3eal33nIcE-|-|-|- sEewHat it$S$a$ys$>>>>>>',
    ...       r'   *f%^&*()%^q*()%^&*m%^&*(l%^&*r%^&r($%^,m,*($9^&@*$%^&*$%^&',
    ...       r'    (p[^rOb]A.bL(y)N0OtH1InG I1Mp-oRt-Ant.iT" "nEvEr1is1!1!    ',
    ...     ]
    >>> out = Trfm.transform(X)
    >>> for line in out:
    ...     print(line)
    THIS IS SOME TERRIBLE TEXT
    DATA CLEAN IT UP REAL NICE
    SEE WHAT IT SAYS PROBABLY
    NOTHING IMPORTANT IT NEVER IS

    """


    def __init__(
        self,
        *,
        universal_sep:Optional[str] = ' ',
        case_sensitive:Optional[bool] = True,
        global_flags:Optional[Union[numbers.Integral, None]] = None,
        remove_empty_rows:Optional[bool] = False,
        join_2D:Optional[str] = ' ',
        return_dim:Optional[ReturnDimType] = None,
        ############
        strip:Optional[bool] = False,
        replace:Optional[ReplaceType] = None,
        remove:Optional[RemoveType] = None,
        normalize:Optional[Union[bool, None]] = False,
        lexicon_lookup:Optional[LexiconLookupType] = None,
        remove_stops:Optional[bool] = False,
        ngram_merge:Optional[NGramsType] = None,
        justify:Optional[Union[numbers.Integral, None]] = None,
        get_statistics:Optional[Union[None, GetStatisticsType]] = None
    ):

        """Initialize the AutoTextCleaner instance."""

        self.universal_sep = universal_sep
        self.case_sensitive = case_sensitive
        self.global_flags = global_flags
        self.remove_empty_rows = remove_empty_rows
        self.join_2D = join_2D
        self.strip = strip
        self.replace = replace
        self.remove = remove
        self.normalize = normalize
        self.lexicon_lookup = lexicon_lookup
        self.remove_stops = remove_stops
        self.ngram_merge = ngram_merge
        self.justify = justify
        self.return_dim = return_dim
        self.get_statistics = get_statistics


        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
        # always initialize these

        self._TJO = TextJoiner(sep=self.join_2D)

        self._TSPL = TextSplitter(
            sep=self.universal_sep,
            case_sensitive=self.case_sensitive,
            maxsplit=None,
            flags=self.global_flags
        )

        # this placement allows _reset to ALWAYS reinstantiate TStat and
        # TextLookupRealTime, clearing any information those classes hold.
        # every reset() should ALWAYS clear the internal state of these
        # submodules.
        # if these were tucked under this (or something like this):
        # if (self.get_statistics or {}).get('before', None) is not None:
        # for conditional instantiation, like they are in transform, they
        # could escape reset when reset is called because user may have
        # set get_statistics/lexicon_lookup to None via set_params in
        # the meantime.
        self._TStatStart = TextStatistics(
            store_uniques=(self.get_statistics or {}).get('before', False) or False
            )

        self._TStatEnd = TextStatistics(
            store_uniques=(self.get_statistics or {}).get('after', False) or False
        )

        self._TL = TextLookupRealTime(
            update_lexicon=(self.lexicon_lookup in ['auto_add', 'manual']),
            skip_numbers=True,
            auto_split=True,
            auto_add_to_lexicon=(self.lexicon_lookup == 'auto_add'),
            auto_delete=(self.lexicon_lookup == 'auto_delete'),
            DELETE_ALWAYS=None,
            REPLACE_ALWAYS=None,
            SKIP_ALWAYS=None,
            SPLIT_ALWAYS=None,
            remove_empty_rows=self.remove_empty_rows,
            verbose=False
        )
        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
        # conditionally initialize these

        if self.strip:
            self._TSTR = TextStripper()

        if self.replace:
            self._TREP = TextReplacer(
                replace=self.replace,
                case_sensitive=self.case_sensitive,
                flags=self.global_flags
            )

        if self.remove:
            self._TREM = TextRemover(
                remove=self.remove,
                case_sensitive=self.case_sensitive,
                remove_empty_rows=self.remove_empty_rows,
                flags=self.global_flags
            )

        if self.normalize is not None:
            self._TN = TextNormalizer(
                upper=self.normalize
            )

        if self.remove_stops:
            self._SR = StopRemover(
                match_callable=None,
                remove_empty_rows=self.remove_empty_rows,
                exempt=None,
                supplemental=None,
                n_jobs=-1    # pizza think on this
            )

        if self.ngram_merge:
            self._NGM = NGramMerger(
                ngrams=self.ngram_merge,
                ngcallable=None,
                sep='_',   # do not use universal_sep here!
                wrap=True,
                case_sensitive=self.case_sensitive,
                remove_empty_rows=self.remove_empty_rows,
                flags=self.global_flags
            )

        if self.justify:
            warnings.warn(
                f"\nYou have selected to justify your data. \nAutoTextCleaner "
                f"will not expose the :attr: `row_support_` attribute."
            )
            self._TJU = TextJustifier(
                n_chars=self.justify,
                sep=self.universal_sep,
                sep_flags=self.global_flags,
                line_break=None,
                line_break_flags=None,
                case_sensitive=self.case_sensitive,
                backfill_sep=self.join_2D,
                join_2D=self.join_2D
            )

        # END conditionally initialize these -- -- -- -- -- -- -- -- --


    def __pybear_is_fitted__(self):
        return True


    @property
    def n_rows_(self):
        """
        Get the `n_rows_` attribute. The total number of rows in
        data passed to :meth: `transform` between resets. This may
        not be the number of rows in the outputted data. Unlike most
        other pybear text transformers that expose an `n_rows_`
        attribute that is not cumulative, this particular attribute
        is cumulative across multiple calls to :meth: `transform`.
        The reason for the different behavior is that the cumulative
        behavior here aligns this attribute with the behavior
        of :attr: `before_statistics_` and :attr: `after_statistics_`,
        which compile statistics cumulatively across multiple calls
        to :meth: `transform`. This number is reset when the
        AutoTextCleaner instance is reset by calls to :meth: `fit`.
        """
        return getattr(self, '_n_rows')


    @property
    def row_support_(self):
        """
        Get the `row_support_` attribute. A 1D boolean numpy vector
        indicating which rows of the data, if any, were removed during
        the cleaning process. The length must equal the number of rows
        in the data originally passed to :meth: `transform`. A row
        that was removed is indicated by a False in the corresponding
        position in the vector, and a row that remains is indicated by
        True. This attribute only reflects the last batch of data passed
        to :meth: `transform`; it is not cumulative.
        """
        return getattr(self, '_row_support')


    @property
    def before_statistics_(self):
        """
        Get the `before_statistics_` attribute. If the 'before' key
        of the :param: `get_statistics` parameter has a value of True
        or False, then statistics about the raw data were compiled
        in a TextStatistics instance before the transformation. This
        exposes that TextStatistics class (which is different from
        the :attr: `after_statistics_` TextStatistics class.) The
        exposed class has attributes that contain information about
        the raw data. See the documentation for TextStatistics to learn
        about what attributes are exposed. The statistics in this
        attribute are reset when the AutoTextCleaner instance is reset
        by calls to :meth: `fit`.
        """
        return getattr(self, '_before_statistics')


    @property
    def after_statistics_(self):
        """
        Get the `after_statistics_` attribute. If the 'after' key of
        the :param: `get_statistics` parameter has a value of True
        or False, then statistics about the transformed data were
        compiled in a TextStatistics instance after the transformation.
        This exposes that TextStatistics class (which is different from
        the :attr: `before_statistics_` TextStatistics class.) The
        exposed class has attributes that contain information about the
        transformed data. See the documentation for TextStatistics to
        learn about what attributes are exposed. The statistics in this
        attribute are reset when the AutoTextCleaner instance is reset
        by calls to :meth: `fit`.
        """
        return getattr(self, '_after_statistics')


    @property
    def lexicon_lookup_(self):
        """
        Get the `lexicon_lookup_` attribute. If :param: `lexicon_lookup`
        has a non-None value, then information about the text-lookup
        process is stored in a TextLookupRealTime (TLRT) instance
        within ATC. This attribute exposes that TLRT class, which has
        attributes that contain information about the handling of
        words not in the pybear Lexicon. If you ran `lexicon_lookup`
        in 'manual' mode, you may have put a lot of effort into handing
        the unknown words and you want access to the information. You
        may have instructed TLRT to hold words that you want to add to
        the Lexicon so that you can access them later and put them in
        the Lexicon. See the documentation for TextLookupRealTime to
        learn about what attributes are exposed. The information in TLRT
        is reset when AutoTextCleaner is reset by calls to :meth: `fit`.
        """
        return getattr(self, '_lexicon_lookup')


    def _reset(self) -> Self:

        """
        Reset the AutoTextCleaner instance. This clears any state
        information that is retained during transformations. This
        includes the :attr: `row_support_` attribute, which holds
        transient state information from the last call to transform.
        This also resets attributes that hold cumulative state
        information, i.e., compiled over many transforms. These
        attributes are the :attr: `n_rows_` counter, and the statistics
        in :attr: `before_statistics_`, :attr: `after_statistics_`,
        and :attr: `lexicon_lookup_`.


        Returns
        -------
        -
            self - the reset AutoTextCleaner instance.

        """

        for _attr in [
            '_n_rows', '_row_support', '_before_statistics',
            '_after_statistics', '_lexicon_lookup'
        ]:
            if hasattr(self, _attr):
                delattr(self, _attr)

        self.__init__(**self.get_params(deep=True))

        return self


    def get_metadata_routing(self):
        raise NotImplementedError(
            f"metadata routing is not implemented in TextRemover"
        )


    # def get_params
    # from GetParamsMixin


    # def set_params
    # from SetParamsMixin


    # def fit_transform
    # from FitTransformMixin


    def partial_fit(
        self,
        X: XContainer,
        y: Optional[Union[any, None]] = None
    ) -> Self:
        """
        No-op batch-wise fitting of the AutoTextCleaner instance.


        Parameters
        ----------
        X:
            XContainer - the 1D or (possibly ragged) 2D text data.
            Ignored.

        y:
            Optional[Union[any, None]], default=None - The target for
            the data. Always ignored.


        Returns
        -------
        -
            self - the AutoTextCleaner instance.

        """


        return self



    def fit(
        self,
        X: XContainer,
        y: Optional[Union[any, None]] = None
    ) -> Self:

        """
        No-op one-shot fitting of the AutoTextCleaner instance.


        Parameters
        ----------
        X:
            XContainer - the 1D or (possibly ragged) 2D text data.
            Ignored.

        y:
            Optional[Union[any, None]], default=None - The target for
            the data. Always ignored.


        Returns
        -------
        -
            self - the AutoTextCleaner instance.

        """

        self._reset()

        return self.partial_fit(X, y)


    def score(
        self,
        X: XContainer,
        y: Optional[Union[any, None]] = None
    ) -> None:

        """
        No-op score method.


        Parameters
        ----------
        X:
            XContainer - the 1D or (possibly ragged) 2D text data.
            Ignored.

        y:
            Optional[Union[any, None]], default=None - The target for
            the data. Always ignored.


        Returns
        -------
        -
            None

        """


        check_is_fitted(self)


        return


    def transform(
        self,
        X: XContainer,
        copy: Optional[bool] = False
    ) -> XWipContainer:

        """
        Process the data as per the parameters.


        Parameters
        ----------
        X:
            XContainer - the 1D or (possibly ragged) 2D text data.

        copy:
            Optional[bool], default=False - Whether to perform the
            text cleaning operations directly on the passed X or on a
            deepcopy of X.


        Returns
        -------
        -
            Union[list[str], list[list[str]] - the processed data.

        """

        _validation(
            X,
            self.universal_sep,
            self.case_sensitive,
            self.global_flags,
            self.remove_empty_rows,
            self.join_2D,
            self.return_dim,
            self.strip,
            self.replace,
            self.remove,
            self.normalize,
            self.lexicon_lookup,
            self.remove_stops,
            self.ngram_merge,
            self.justify,
            self.get_statistics
        )

        if copy:
            _X = _copy_X(X)
        else:
            _X = X


        _X = _map_X_to_list(_X)

        self._n_rows = getattr(self, '_n_rows', 0) + len(_X)

        # do not use _n_rows! it is cumulative.
        self._row_support = np.ones(len(_X), dtype=bool)


        if (self.get_statistics or {}).get('before', None) is not None:
            # this class is always available, only fit if it is turned on
            self._TStatStart.partial_fit(_X)
            # declaring this dummy controls access via @property. The TS
            # start & end instances are ALWAYS (not conditionally)
            # instantiated in init to guarantee exposure whenever reset()
            # is called. They are only actually USED when the 'if' allows
            # it. So grant access to them via @property only when they
            # are actually USED.
            self._before_statistics = self._TStatStart


        # do this before shape-shift
        if self.strip:
            # example axis cannot change
            _X = self._TSTR.transform(_X, copy=False)


        # pre-processing shaping -- -- -- -- -- -- -- -- -- -- -- -- --
        # if there are inputs for lexicon_lookup, remove_stops, or ngram_merge,
        # those MUST have 2D inputs (and outputs) so if _X is passed as 1D,
        # must convert to 2D. (Then later deal with whatever return_dim is.)
        _was_1D, _is_1D = False, False
        if all(map(isinstance, _X, (str for _ in _X))):
            _was_1D, _is_1D = True, True

        if any((self.lexicon_lookup, self.remove_stops, self.ngram_merge)) \
                and _is_1D:

            _X = self._TSPL.transform(_X)
            # do not change _was_1D
            _is_1D = False
        # END pre-processing shaping -- -- -- -- -- -- -- -- -- -- -- --


        # this is where strip was before moving it to the top


        if self.replace:
            # example axis cannot change
            _X = self._TREP.transform(_X, copy=False)


        if self.remove:
            # has remove_empty_rows, example axis can change
            _X = self._TREM.transform(_X)
            # len(TREM.row_support) must == number of Trues in self._row_support
            assert len(self._TREM.row_support_) == sum(self._row_support)
            # whatever changed in the currently outputted row_support_ only
            # impacts the entries in self._row_support that are True
            self._row_support[self._row_support] = self._TREM.row_support_


        if self.normalize is not None:
            # example axis cannot change
            _X = self._TN.transform(_X)


        if self.lexicon_lookup:
            # has remove_empty_rows, example axis can change
            _X = self._TL.transform(_X)
            # this class is always available, only transforms if it is turned on
            # declaring this dummy controls access via @property. The
            # TextLookupRealTime instance is ALWAYS (not conditionally)
            # instantiated in init to guarantee exposure whenever reset()
            # is called. It is only actually USED when the 'lexicon_lookup'
            # parameter is not None. So grant access to them via @property
            # only when it is actually USED.
            self._lexicon_lookup = self._TL
            # len(TL.row_support) must == number of Trues in self._row_support
            assert len(self._TL.row_support_) == sum(self._row_support)
            # whatever changed in the currently outputted row_support_ only
            # impacts the entries in self._row_support that are True
            self._row_support[self._row_support] = self._TL.row_support_


        if self.remove_stops:
            # has remove_empty_rows, example axis can change
            _X = self._SR.transform(_X)
            # len(TL.row_support) must == number of Trues in self._row_support
            assert len(self._SR.row_support_) == sum(self._row_support)
            # whatever changed in the currently outputted row_support_ only
            # impacts the entries in self._row_support that are True
            self._row_support[self._row_support] = self._SR.row_support_


        if self.ngram_merge:
            # has remove_empty_rows, example axis can change
            _X = self._NGM.transform(_X)
            # len(TL.row_support) must == number of Trues in self._row_support
            assert len(self._NGM.row_support_) == sum(self._row_support)
            # whatever changed in the currently outputted row_support_ only
            # impacts the entries in self._row_support that are True
            self._row_support[self._row_support] = self._NGM.row_support_


        if self.justify:
            _X = self._TJU.transform(_X)
            warnings.warn(
                f"\nAutoTextCleaner will not expose the :attr: `row_support_` "
                f"attribute because 'justify' is enabled."
            )
            self._row_support = None


        if (self.get_statistics or {}).get('after', None) is not None:
            # this class is always available, only fit if it is turned on
            self._TStatEnd.partial_fit(_X)
            # declaring this dummy controls access via @property. The TS
            # start & end instances are ALWAYS (not conditionally)
            # instantiated in init to guarantee exposure whenever reset()
            # is called. They are only actually USED when the 'if' allows
            # it. So grant access to them via @property only when they
            # are actually USED.
            self._after_statistics = self._TStatEnd


        # final shaping -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

        _need_1D, _need_2D = False, False
        if self.return_dim is None:
            # need to return in same dim as given
            if _was_1D and not _is_1D:
                _need_1D = True
            elif not _was_1D and _is_1D:
                _need_2D = True
        elif self.return_dim == '1D' and not _is_1D:
            _need_1D = True
        elif self.return_dim == '2D' and _is_1D:
            _need_2D = True

        assert not (_need_1D and _need_2D)

        if _need_1D:
            _X = self._TJO.transform(_X)
        elif _need_2D:
            _X = self._TSPL.transform(_X)

        # END final shaping -- -- -- -- -- -- -- -- -- -- -- -- -- -- --


        return _X





