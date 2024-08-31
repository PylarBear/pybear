# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



"""

    refit:
        bool, str, or callable, default=True - After completion of the
        grid search, fit the estimator on the whole dataset using the
        best found parameters, and expose this fitted estimator via the
        best_estimator_ attribute. Also, when the estimator is refit the
        GSTCV instance itself becomes the best estimator, exposing the
        predict_proba, predict, and score methods (and possibly others.)
        When refit is not performed, the search simply finds the best
        parameters and exposes them via the best_params_ attribute
        (unless there are multiple scorers and refit is False, in which
        case information about the grid search is only available via the
        cv_results_ attribute.)

        The values accepted by refit depend on the scoring scheme, that
        is, whether a single or multiple scorers are used. In all cases,
        refit can be boolean False (to disable refit), a string that
        indicates the scorer to use to determine the best parameters
        (when there is only one scorer there is only one possible string
        value), or a callable. See below for more information about the
        refit callable. When one metric is used, refit can be boolean
        True or False, but boolean True cannot be used when there is
        more than one scorer.

        Where there are considerations other than maximum score in
        choosing a best estimator, refit can be set to a function that
        takes in cv_results_ and returns the best_index_ (an integer).
        In that case, best_params_ and best_estimator_ will be set
        according to the returned best_index_. The best_score_ and
        best_threshold_ attributes will not be available if there are
        multiple scorers, but are available if there is only one scorer.

        See scoring parameter to know more about multiple metric
        evaluation.

"""















