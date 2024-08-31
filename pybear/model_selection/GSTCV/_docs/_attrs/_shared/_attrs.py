# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



# cv_results_ - always exposed after fit()
# best_estimator_ - only exposed when refit is not False
# best_score_ - always exposed when there is one scorer, or when refit is a str for 2+ scorers
# best_params_ - always exposed when there is one scorer, or when refit is not False for 2+ scorers
# best_index_ - always exposed when there is one scorer, or when refit is not False for 2+ scorers
# scorer_ - always exposed after fit()
# n_splits_ - always exposed after fit()
# refit_time_ - only exposed when refit is not False
# multimetric_ - always exposed after fit()
# classes_ - only exposed when refit is not False
# n_features_in_ - only exposed when refit is not False
# feature_names_in_ - only exposed when refit is not False and a DF is passed to fit()
# best_threshold_- always exposed when there is one scorer, or when refit is a str for 2+ scorers



"""

    cv_results_:
        dict[str, np.ma.maskedarray] - A dictionary with column headers
        as keys and results as values, that can be conveniently converted
        into a pandas DataFrame.

        Always exposed after fit.

        Below is an example of cv_results_ for a logistic classifier,
        with:
            cv=3,
            param_grid={'C': [1e-5, 1e-4]},
            thresholds=np.linspace(0,1,21),
            scoring=['accuracy', 'balanced_accuracy']
            return_train_score=False

        on random data.

        {
            'mean_fit_time':                    [1.227847, 0.341168]
            'std_fit_time':                     [0.374309, 0.445982]
            'mean_score_time':                  [0.001638, 0.001676]
            'std_score_time':                   [0.000551, 0.000647]
            'param_C':                             [0.00001, 0.0001]
            'params':                  [{'C': 1e-05}, {'C': 0.0001}]
            'best_threshold_accuracy':                   [0.5, 0.51]
            'split0_test_accuracy':              [0.785243, 0.79844]
            'split1_test_accuracy':              [0.80228, 0.814281]
            'split2_test_accuracy':             [0.805881, 0.813381]
            'mean_test_accuracy':               [0.797801, 0.808701]
            'std_test_accuracy':                [0.009001, 0.007265]
            'rank_test_accuracy':                             [2, 1]
            'best_threshold_balanced_accuracy':          [0.5, 0.51]
            'split0_test_balanced_accuracy':    [0.785164, 0.798407]
            'split1_test_balanced_accuracy':    [0.802188, 0.814252]
            'split2_test_balanced_accuracy':    [0.805791, 0.813341]
            'mean_test_balanced_accuracy':      [0.797714, 0.808667]
            'std_test_balanced_accuracy':       [0.008995, 0.007264]
            'rank_test_balanced_accuracy':                    [2, 1]
        }

        Slicing across the masked arrays yields the results for the fit
        and score of a single set of search points. That is, indexing
        into all of the masked arrays at position zero yields the result
        for the first set of search points, index 1 contains the results
        for the second set of points, and so forth.

        The key 'params' is used to store a list of parameter settings
        dicts for all the parameter candidates. That is, the 'params'
        key holds all the possible permutations of parameters for the
        given search grid(s).

        The mean_fit_time, std_fit_time, mean_score_time and
        std_score_time are all in seconds.

        For single-metric evaluation, the scores for the single scorer
        are available in the cv_results_ dict at the keys ending with
        '_score'. For multi-metric evaluation, the scores for all the
        scorers are available in the cv_results_ dict at the keys ending
        with that scorer’s name ('_<scorer_name>').
        (‘split0_test_precision’, ‘mean_train_precision’ etc.)

    best_estimator_:
        estimator - The estimator that was chosen by the search, i.e.
        the estimator which gave highest score (or smallest loss if
        specified) on the held-out (test) data. Only exposed when refit
        is not False; see refit parameter for more information on allowed
        values.

    best_score_:
        float - The mean of the scores of the hold out (test) cv folds
        for the best estimator. Always exposed when there is one scorer,
        or when refit is specified as a string for 2+ scorers.

    best_params_:
        dict[str, any] - Exposes the dictionary found at
        cv_results_['params'][best_index_], which gives the parameter
        settings that resulted in the highest mean score (best_score_)
        on the hold out (test) data.

        best_params_ never holds best_threshold_. Access best_threshold_
        via the best_threshold_ attribute (if available) or the
        cv_results_ attribute.

        best_params_ is always exposed when there is one scorer, or when
        refit is not False for 2+ scorers.

    best_index_:
        int - The index of the cv_results_ arrays which corresponds to
        the best parameter settings. Always exposed when there is one
        scorer, or when refit is not False for 2+ scorers.

    scorer_:
        dict - Scorer metric(s) used on the held out data to choose the
        best parameters for the model. Always exposed after fit.

        This attribute holds the validated scoring dictionary which maps
        the scorer key to the scorer metric callable, i.e., a dictionary
        of {scorer_name: scorer_metric}.

    n_splits_:
        int -  The number of cross-validation splits (folds/iterations).
        Always exposed after fit.

    refit_time_:
        float - Seconds elapsed when refitting the best model on the
        whole dataset. Only exposed when refit is not False.

    multimetric_:
        bool - Whether or not several scoring metrics were used. False
        if one scorer was used, otherwise True. Always exposed after fit.

    classes_:
        ndarray of shape (n_classes,) - Class labels. Only exposed when
        refit is not False. Because GSTCV imposes a restriction that y
        must be binary in [0, 1], this must always return [0, 1].

    feature_names_in_:
        ndarray of shape (n_features_in_,) - Names of features seen
        during fit.

        Only exposed when refit is not False (see the documentation for
        the refit parameter for more details) and a dataframe was passed
        to fit.

    best_threshold_:
        float - The threshold that, when used along with the parameter
        values found in best_params_, yields the highest score for the
        given settings and data.

        When one scorer is used, the best threshold found is always
        exposed via the best_threshold_ attribute. When multiple scorers
        are used, the best_threshold_ attribute is only exposed when a
        string value is passed to the refit kwarg.

        The best threshold is only available conditionally via the
        best_threshold_ attribute. Another way to discover the best
        threshold for each scorer is by inspection of the cv_results_
        attribute.


"""



