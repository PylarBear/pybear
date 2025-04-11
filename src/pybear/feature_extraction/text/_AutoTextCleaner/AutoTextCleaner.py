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
                wrap=True,
                case_sensitive=self.case_sensitive,
                remove_empty_rows=self.remove_empty_rows,
                flags=self.global_flags
            )


        if self.justify:

            warnings.warn(
                f"You have selected to justify your data. \nAutoTextCleaner "
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
        return getattr(self, '_n_rows')


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
        return getattr(self, '_row_support')


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

        self._n_rows = len(_X)


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


        self._row_support = np.ones(self._n_rows, dtype=bool)


        if (self.get_statistics or {}).get('before', None) is not None:
            self._TStatStart.partial_fit(_X)


        # this is where strip was before moving it to the top


        if self.replace:
            # example axis cannot change
            _X = self._TREP.transform(_X, copy=False)


        if self.remove:
            # has remove_empty_rows, example axis can change
            _X = self._TREM.transform(_X)
            # the length of TL.row_support must == number of Trues in self._row_support
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


        if self.justify:
            _X = self._TJU.transform(_X)
            warnings.warn(
                f"AutoTextCleaner will not expose the :attr: `row_support_` "
                f"attribute because 'justify' is active."
            )
            self._row_support = None


        if (self.get_statistics or {}).get('after', None) is not None:
            self._TStatEnd = TextStatistics(
                store_uniques=self.get_statistics['after']
            )

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





