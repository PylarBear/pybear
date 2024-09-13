# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


"""

    thresholds:
        Union[None, Union[int, float], vector-like[Union[int, float]] -
        The decision threshold search grid to use when performing hyper-
        parameter search. Other GridSearchCV modules only allow for
        search at the conventional decision threshold for binary class-
        ifiers, 0.5. This module additionally searches over any set of
        decision threshold values in the 0 to 1 interval (inclusive) in
        the same manner as any other hyperparameter while performing the
        grid search.

        The thresholds parameter can be passed via the 'thresholds'
        kwarg. In this case, thresholds can be None, a single number
        from 0 to 1 (inclusive) or a list-like of such numbers. If None,
        (and thresholds are not passed directly inside the param grid(s)),
        the default threshold grid is used, numpy.linspace(0, 1, 21).

        Thresholds may also be passed to individual param grids via a
        'thresholds' key. However, when passed directly to a param grid,
        thresholds cannot be None or a single number, it must be a list-
        like of numbers as is normally done with param grids.

        Because 'thresholds' can be passed in 2 different ways, there is
        a hierarchy that dictates which thresholds are used during
        searching and scoring. Any threshold values passed directly
        within a param grid always supersede any passed (or not passed)
        to the 'thresholds' kwarg. When no thresholds are passed inside
        a param grid, the values passed as a kwarg are used -- if no
        values were passed as a kwarg, then the default values are
        used. If all passed param grids have no 'thresholds' entry, then
        whatever is passed to the kwarg is used for all of them; if the
        'thresholds' kwarg is left as default, then the default threshold
        grid is used for all the grids.

        When one scorer is used, the best threshold is always exposed
        and is accessible via the best_threshold_ attribute. When
        multiple scorers are used, the best_threshold_ attribute is only
        exposed when a string value is passed to the refit kwarg.
        The best threshold is never reported in the best_params_
        attribute, even if thresholds were passed via a param grid; the
        best threshold is only available conditionally via the
        best_threshold_ attribute. Another way to discover the best
        threshold for each scorer is by inspection of the cv_results_
        attribute.

        The scores reported for test data in cv_results_ are those for
        the best threshold. Also note that when return_train_score is
        True, the scores returned for the train data are only for the
        best threshold found for the test data. That is, the thresholds
        are scored for the test data, the best score is found, and the
        best threshold is set based on the threshold for that score.
        Then when scoring train data, only that threshold is scored and
        reported in cv_results_.









"""









