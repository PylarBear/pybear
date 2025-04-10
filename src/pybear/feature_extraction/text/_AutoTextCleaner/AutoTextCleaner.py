# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Literal, Optional, Sequence
from typing_extensions import Self, TypeAlias, Union
from ._type_aliases import (
    XContainer,
    XWipContainer,
    MatchType,
    ReplaceType,
    GetStatisticsType
)

import numbers
import re

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



    Parameters
    ----------


    Attributes
    ----------


    Notes
    -----
    Type Aliases



    Examples
    --------


    """

    def __init__(
        self,
        *,
        universal_sep:Optional[str] = ' ',
        case_sensitive:Optional[bool] = True,
        remove_empty_rows:Optional[bool] = False,
        join_2D:Optional[str] = ' ',
        return_dim:Optional[Union[Literal['1D', '2D'], None]] = None,
        ############
        strip:Optional[bool] = True,
        replace:Optional[Union[tuple[MatchType, ReplaceType], None]] = (re.compile('[^a-z0-9]', re.I), ''),
        remove:Optional[Union[MatchType, list[MatchType], None]] = re.compile('^[^a-z0-9]+$', re.I),
        normalize:Optional[Union[bool, None]] = True,
        lexicon_lookup:Optional[Union[Literal['auto_add', 'auto_delete', 'manual'], None]] = 'auto_delete',
        remove_stops:Optional[bool] = True,
        ngram_merge:Optional[Union[Sequence[tuple[MatchType, ...]], None]] = None,
        justify:Optional[Union[numbers.Integral, None]] = None,
        get_statistics:Optional[Union[None, GetStatisticsType]] = None
    ):

        """Initialize the AutoTextCleaner instance."""

        self.universal_sep = universal_sep
        self.case_sensitive = case_sensitive
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
            flags=None
        )
        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
        # conditionally initialize these
        if (self.get_statistics or {}).get('before', None) is not None:
            self._TStatStart = TextStatistics(
                store_uniques=self.get_statistics['before']
            )


        if self.strip:
            self._TSTR = TextStripper()


        if self.replace:
            self._TREP = TextReplacer(
                replace=self.replace,
                case_sensitive=self.case_sensitive,
                flags=None
            )


        if self.remove:
            self._TREM = TextRemover(
                remove=self.remove,
                case_sensitive=self.case_sensitive,
                remove_empty_rows=self.remove_empty_rows,
                flags=None
            )


        if self.normalize is not None:
            self._TN = TextNormalizer(
                upper=self.normalize
            )


        # Optional[Union[Literal['auto_add', 'auto_delete', 'manual'], None]] = 'auto_delete'
        if self.lexicon_lookup:
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
                wrap=False,
                case_sensitive=self.case_sensitive,
                remove_empty_rows=self.remove_empty_rows,
                flags=None
            )


        if self.justify:
            self._TJU = TextJustifier(
                n_chars=self.justify,
                sep=self.universal_sep,
                sep_flags=None,
                line_break=None,
                line_break_flags=None,
                case_sensitive=self.case_sensitive,
                backfill_sep=self.join_2D,
                join_2D=self.join_2D
            )


        if (self.get_statistics or {}).get('after', None) is not None:
            self._TStatEnd = TextStatistics(
                store_uniques=self.get_statistics['after']
            )

        # END conditionally initialize these -- -- -- -- -- -- -- -- --


    def __pybear_is_fitted__(self):
        return True


    @property
    def n_rows_(self):
        """
        Get the `n_rows_` attribute. The number of rows in the data
        passed to :meth: `transform`. This may not be the number of
        rows in the outputted data. Not cumulative, only reflects the
        last batch of data passed to :meth: `transform`.

        """
        return


    @property
    def row_support_(self):
        """
        Get the `row_support_` attribute. A 1D boolean numpy vector
        indicating which rows of the data, if any, were removed during
        the cleaning process. The length must equal the number of rows
        in the data originally passed to :meth: `transform`, and must
        also equal :attr: `n_rows_`. A row that was removed is indicated
        by a False in the corresponding position in the vector, and a
        row that remains is indicated by True.

        """
        # pizza
        # return self._row_support



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

        # pizza _validation

        if copy:
            _X = _copy_X(X)
        else:
            _X = X


        _X = _map_X_to_list(_X)

        # pre-processing shaping -- -- -- -- -- -- -- -- -- -- -- -- --
        # if there are inputs for lexicon_lookup, remove_stops, or ngram_merge,
        # those MUST have 2D inputs (and outputs) so if _X is passed as 1D,
        # must convert to 2D. (Then later deal with whatever return_dim is.)
        _is_1D = False
        if all(map(isinstance, _X, (str for _ in _X))):
            _is_1D = True

        if any((self.lexicon_lookup, self.remove_stops, self.ngram_merge)) \
                and _is_1D:

            _X = self._TJO.transform(_X)
            _is_1D = False
        # END pre-processing shaping -- -- -- -- -- -- -- -- -- -- -- --



        if (self.get_statistics or {}).get('before', None) is not None:
            self._TStatStart.partial_fit(_X)


        if self.strip:
            # example axis cannot change
            _X = self._TSTR.transform(_X, copy=False)


        if self.replace:
            # example axis cannot change
            _X = self._TREP.transform(_X, copy=False)


        if self.remove:
            # has remove_empty_rows, example axis can change
            _X = self._TREM.transform(_X)
            self._row_support = self._TREM.row_support_


        if self.normalize is not None:
            # example axis cannot change
            _X = self._TN.transform(_X)


        if self.lexicon_lookup:
            # has remove_empty_rows, example axis can change
            _X = self._TL.transform(_X)
            # the length of TL.row_support must == number of Trues in self._row_support
            assert len(self._TL.row_support_) == sum(self._row_support)
            # whatever changed in the currently outputted row_support_ only
            # impacts the entries in self._row_support that are True
            self._row_support[self._row_support] = self._TL.row_support_


        if self.remove_stops:
            # has remove_empty_rows, example axis can change
            _X = self._SR.transform(_X)
            # the length of TL.row_support must == number of Trues in self._row_support
            assert len(self._SR.row_support_) == sum(self._row_support)
            # whatever changed in the currently outputted row_support_ only
            # impacts the entries in self._row_support that are True
            self._row_support[self._row_support] = self._SR.row_support_


        if self.ngram_merge:
            # has remove_empty_rows, example axis can change
            _X = self._NGM.transform(_X)
            # the length of TL.row_support must == number of Trues in self._row_support
            assert len(self._NGM.row_support_) == sum(self._row_support)
            # whatever changed in the currently outputted row_support_ only
            # impacts the entries in self._row_support that are True
            self._row_support[self._row_support] = self._NGM.row_support_


        # pizza, if this is true it blows away the meaning of row_support_
        if self.justify:
            _X = self._TJU.transform(_X)


        if (self.get_statistics or {}).get('after', None) is not None:
            self._TStatEnd = TextStatistics(
                store_uniques=self.get_statistics['after']
            )

        # final shaping -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
        if self.return_dim is None:
            pass
        elif _is_1D and self.return_dim == '1D':
            pass
        elif not _is_1D and self.return_dim == '2D':
            pass
        elif _is_1D and self.return_dim == '2D':
            _X = self._TSPL.transform(_X)
        elif not _is_1D and self.return_dim == '1D':
            _X = self._TJO.transform(_X)
        # END final shaping -- -- -- -- -- -- -- -- -- -- -- -- -- -- --


        return _X





