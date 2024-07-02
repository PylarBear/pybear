# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import scipy.stats as ss

from ..._type_aliases import (
    CVResultsType,
    ScorerWIPType
)



def _cv_results_rank_update(
    _scorer: ScorerWIPType,
    _cv_results: CVResultsType
    ) -> CVResultsType:


    for scorer_suffix in _scorer:

        if f'rank_test_{scorer_suffix}' not in _cv_results:
            raise ValueError(f"appending tests scores to a column in cv_results_ "
                f"that doesnt exist but should (rank_test_{scorer_suffix})")

        _ = _cv_results[f'mean_test_{scorer_suffix}']
        _cv_results[f'rank_test_{scorer_suffix}'] = \
            len(_) - ss.rankdata(_, method='max') + 1
        del _

    return _cv_results











