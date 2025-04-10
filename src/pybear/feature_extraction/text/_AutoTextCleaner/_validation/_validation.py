# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing_extensions import Union
from .._type_aliases import (
    XContainer,
    ReturnDimType,
    GetStatisticsType,
    LexiconLookupType
)

import numbers

from ...__shared._validation._1D_2D_X import _val_1D_2D_X
from ...__shared._validation._any_bool import _val_any_bool
from ...__shared._validation._any_integer import _val_any_integer
from ...__shared._validation._any_string import _val_any_string


from ._get_statistics import _val_get_statistics
from ._lexicon_lookup import _val_lexicon_lookup
from ._return_dim import _val_return_dim




# pizza, maybe a warning that if 'justify' is not None, then row_support_
# will not be available.


def _validation(
    _X: XContainer,
    _universal_sep: str,
    _case_sensitive: bool,
    _remove_empty_rows: bool,
    _join_2D: str,
    _return_dim: ReturnDimType,
    _strip: bool,
    _repace,   # pizza
    _remove,   # pizza
    _normalize: Union[bool, None],
    _lexicon_lookup: LexiconLookupType,
    _remove_stops: bool,
    _justify: Union[numbers.Integral, None],
    _get_statistics: GetStatisticsType
) -> None:


    """
    Validate the parameters for AutoTextCleaner. The brunt of the
    validation is handled by the submodules. See them for more
    information. This is a centralized hub for all the submodules.


    Parameters
    ----------
    _strip


    Returns
    -------
    -
        None


    """


    _val_1D_2D_X(_X)
    _val_any_string(_universal_sep, 'universal_sep', _can_be_None=False)
    _val_any_bool(_case_sensitive, 'case_sensitive', _can_be_None=False)
    _val_any_bool(_remove_empty_rows, 'remove_empty_rows', _can_be_None=False)
    _val_any_string(_join_2D, 'join_2D', _can_be_None=False)
    _val_return_dim(_return_dim)
    # ############
    _val_any_bool(_strip, 'strip', _can_be_None=False)
    # replace:Optional[Union[tuple[MatchType, ReplaceType], None]] = (re.compile('[^a-z0-9]', re.I), ''),
    # remove:Optional[Union[MatchType, list[MatchType], None]] = re.compile('^[^a-z0-9]+$', re.I),
    _val_any_bool(_normalize, 'normalize', _can_be_None=True)
    _val_lexicon_lookup(_lexicon_lookup)
    _val_any_bool(_remove_stops, 'remove_stops', _can_be_None=False)
    # ngram_merge:Optional[Union[Sequence[tuple[MatchType, ...]], None]] = None,
    _val_any_integer(_justify, 'justify', _can_be_bool=False, _can_be_None=True)
    _val_get_statistics(_get_statistics)










