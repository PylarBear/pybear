# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Callable
from typing_extensions import Union
import numpy.typing as npt

import numbers

import numpy as np
from joblib import Parallel, delayed, wrap_non_picklable_objects



def _transform(
    _X: list[list[str]],
    _callable: Callable[[str, str], bool],
    _stop_words: list[str],
    _remove_empty_rows: bool,
    _n_jobs: Union[numbers.Integral, None]
) -> tuple[list[list[str]], npt.NDArray[bool]]:

    """
    Remove stop words from X. If required, remove any rows made empty by
    the stop word removal process. Return a boolean numpy vector
    indicating which rows were kept (True) after the empty row removal
    process.


    Parameters
    ----------
    _X:
        list[list[str]] - the text data.
    _callable:
        Callable[[str, str], bool] - the umpire function for determining
        if a token in X is a match against a stop word.
    _stop_words:
        list[str] - the list of stop words from pybear.Lexicon
    _remove_empty_rows:
        bool - Whether to remove any empty rows that might result from
        the stop word removal process.
    _n_jobs:
        Union[numbers.Integral, None] - the number of rows to search in
        parallel for stop words.


    Returns
    -------
    -
        X: tuple[list[list[str]], npt.NDArray[bool]] - the data with
        stop words removed, a boolean numpy vector indicating with rows
        were kept from the original data (True).


    """


    # parallel helper function -- -- -- -- -- -- -- -- -- -- -- -- -- --
    @wrap_non_picklable_objects
    def _parallel_matcher(
        _callable: Callable[[str, str], bool],
        _line: list[str],
        _stop_words: list[str]
    ) -> list[str]:

        """
        Parallelized function for finding stop words in one row of X.


        Parameters
        ----------
        _callable:
            the umpire function that determines if a word in X is a stop
            word.
        _line:
            list[str] - a single line in X; the line currently being
            searched for stop words and having them removed.
        _stop_words:
            list[str] - the list of stop words from pybear.Lexicon.


        Returns
        -------
        -
            list[str]: a single row of X with stop words removed.

        """

        _MASK = np.zeros((len(_line), )).astype(np.uint8)
        for _sw in _stop_words:
            _MASK += np.fromiter(
                map(_callable, _line, (_sw for _ in _line)),
                dtype=np.uint8
            )

        out = np.array(_line)[np.logical_not(_MASK.astype(bool))].tolist()

        del _line, _MASK, _sw

        return out
    # END _parallel_matcher -- -- -- -- -- -- -- -- -- -- -- -- --


    _joblib_kwargs = {'n_jobs': _n_jobs, 'return_as': 'list', 'prefer': 'processes'}
    _X = Parallel(**_joblib_kwargs)(
        delayed(_parallel_matcher)(_callable, _line, _stop_words) for _line in _X
    )


    _row_support: npt.NDArray[bool] = np.ones((len(_X),)).astype(bool)
    for r_idx in range(len(_X) - 1, -1, -1):

        if _remove_empty_rows and len(_X[r_idx]) == 0:
            _row_support[r_idx] = False
            _X.pop(r_idx)


    return _X, _row_support



















