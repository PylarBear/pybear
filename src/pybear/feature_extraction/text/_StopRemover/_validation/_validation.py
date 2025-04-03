# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Callable, Sequence
from typing_extensions import Union
from .._type_aliases import XContainer

import numbers

from ._match_callable import _val_match_callable

from ._exempt import _val_exempt

from ._supplemental import _val_supplemental

from ._n_jobs import _val_n_jobs

from ...__shared._validation._2D_X import _val_2D_X
from ...__shared._validation._any_bool import _val_any_bool



def _validation(
    _X: XContainer,
    _match_callable: Union[Callable[[str, str], bool], None],
    _remove_empty_rows: bool,
    _exempt: Union[Sequence[str], None],
    _supplemental: Union[Sequence[str], None],
    _n_jobs: Union[numbers.Integral, None]
) -> None:

    """
    Centralized hub for validating parameters for StopRemover. The
    brunt of validation is handled by the individual validation modules.
    See the individual modules for more details.


    Parameters
    ----------
    _X:
        XContainer - the data from which to remove stop words.
    _match_callable:
        Union[Callable[[str, str], bool], None] - None to use the default
        StopRemover matching criteria, or a custom callable that defines
        what constitutes matches of words in the text against the stop
        words.
    _remove_empty_rows:
        bool - whether to remove any empty rows that may be left after
        the stop word removal process.
    _exempt:
        Optional[Union[list[str], None]] - stop words that are exempted
        from the search. text that matches these words will not be
        removed.
    _supplemental:
        Optional[Union[list[str], None]] - words to be removed in
        addition to the stop words.
    _n_jobs:
        Optional[Union[numbers.Integral, None]], default = -1 - the
        number of cores/threads to use when parallelizing the search for
        stop words in the rows of X. The default is to use processes but
        can be set by running StopRemover under a joblib parallel_config
        context manager. -1 uses all available cores/threads. None uses
        joblib's default number of cores/threads.


    Returns
    -------
    -
        None

    """


    _val_2D_X(_X, _require_all_finite=False)

    _val_match_callable(_match_callable)

    _val_any_bool(_remove_empty_rows, 'remove_empty_rows')

    _val_exempt(_exempt)

    _val_supplemental(_supplemental)

    _val_n_jobs(_n_jobs)







