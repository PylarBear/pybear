# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#




from typing import Literal, Optional, Sequence
from typing_extensions import Self, TypeAlias, Union
from ._type_aliases import (
    XContainer,
    XWipContainer
)

import numbers
import re

from .._NGramMerger.NGramMerger import NGramMerger
from .._StopRemover.StopRemover import StopRemover
from .._TextJoiner.TextJoiner import TextJoiner
from .._TextJustifier.TextJustifier import TextJustifier
from .._TextLookup.TextLookup import TextLookup
from .._TextNormalizer.TextNormalizer import TextNormalizer
from .._TextPadder.TextPadder import TextPadder
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



MatchType: TypeAlias = Union[str, re.Pattern]


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


    """

    def __init__(
        self,
        *,
        universal_sep:Optional[str] = ' ',
        universal_case_sensitive:Optional[bool] = True,
        universal_delete_empty_rows:Optional[bool] = False,
        strip:Optional[bool] = True,
        replace:Optional[Union[dict[MatchType, str], None]] = {re.compile('[^a-z0-9]', re.I): ''},
        remove:Optional[Union[MatchType, list[MatchType], None]] = re.compile('^[^a-z0-9]+$', re.I),
        normalize:Optional[Union[bool, None]] = True,
        lexicon_lookup:Optional[Union[Literal['auto_add', 'auto_delete'], None]] = 'auto_delete',
        remove_stops:Optional[bool] = True,
        ngram_merge:Optional[Union[Sequence[tuple[MatchType, ...]], None]] = None,
        justify:Optional[Union[numbers.Integral, None]] = False,
        return_dim:Optional[Union[Literal['1D', '2D'], None]] = None,
        get_statistics: Optional[bool] = False
    ):

        """Initialize the AutoTextCleaner instance."""

        self.universal_sep = universal_sep
        self.universal_case_sensitive = universal_case_sensitive
        self.universal_delete_empty_rows = universal_delete_empty_rows
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


    def __pybear_is_fitted__(self):
        return True


    @property
    def n_rows_(self):
        """


        """
        return


    @property
    def row_support_(self):
        """


        """
        return



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

        # pizza _validation

        if copy:
            _X = _copy_X(X)
        else:
            _X = X


        _X = _map_X_to_list(_X)


        #         universal_sep:Optional[str] = ' ',
        #         strip:Optional[bool] = True,
        #         replace:Optional[Union[dict[MatchType, str], None]] = {re.compile('[^a-z0-9]', re.I): ''},
        #         remove:Optional[Union[MatchType, list[MatchType], None]] = re.compile('^[^a-z0-9]+$', re.I),
        #         normalize:Optional[Union[bool, None]] = True,
        #         lexicon_lookup:Optional[Union[Literal['auto_add', 'auto_delete'], None]] = 'auto_delete',
        #         remove_stops:Optional[bool] = True,
        #         ngram_merge:Optional[Union[Sequence[tuple[MatchType, ...]], None]] = None,
        #         justify:Optional[Union[numbers.Integral, None]] = False,
        #         return_dim:Optional[Union[Literal['1D', '2D'], None]] = None,
        #         delete_empty_rows:Optional[bool] = False,
        #         get_statistics: Optional[bool] = False


        NGM = NGramMerger(ngrams=self.ngram_merge)
        SR = StopRemover()
        TJ = TextJoiner()
        TJRE = TextJustifier()
        TL = TextLookup()
        TN = TextNormalizer()
        TP = TextPadder()
        TRem = TextRemover()
        TRep = TextReplacer()
        TSpl = TextSplitter()
        TStat = TextStatistics()
        TStr = TextStripper()


        return _X





