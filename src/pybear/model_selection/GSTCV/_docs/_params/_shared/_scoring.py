# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



"""
    
    scoring:
        Union[str, callable, vector-like[str], dict[str, callable]],
        default='accuracy' - Strategy to evaluate the performance of the
        cross-validated model on the test set (and also train set, if
        return_train_score is True.)

        For any number of scorers, scoring can be a dictionary with user-
        assigned scorer names as keys and callables as values. See below
        for clarification on allowed callables.

        For a single scoring metric, a single string or a single callable
        is also allowed. Valid strings that can be passed are 'accuracy',
        'balanced_accuracy', 'average_precision', 'f1', 'precision', and
        'recall'.

        For evaluating multiple metrics, scoring can also be a vector-
        like of unique strings, containing a combination of the allowed
        strings.

        The default scorer of the estimator cannot used by this module
        because the decision threshold cannot be manipulated. Therefore,
        'scoring' cannot accept a None argument.

        About the scorer callable:
        This module's scorers deviate from other GridSearch implement-
        ations in an important way. Some of those implementations accept
        make_scorer functions, e.g. sklearn.metrics.make_scorer, but
        this module cannot accept this. make_scorer implicitly assumes a
        decision threshold of 0.5, but this module needs to be able to
        calculate predictions based on any user-entered threshold. There-
        fore, in place of make_scorer functions, this module uses scoring
        metrics directly (whereas they would otherwise be passed to
        make_scorer.)

        Additionally, this module can accept any scoring function that
        has signature (y_true, y_pred) and returns a single number. Note
        that, when using a custom scorer, the scorer should return a
        single value. Metric functions returning a list/array of values
        can be wrapped into multiple scorers that return one value each.

        This module cannot directly accept scorer kwargs and pass them
        to scorers. To pass kwargs to your scoring metric, create a
        wrapper with signature (y_true, y_pred) around the metric and
        hard-code the kwargs into the metric, e.g.,

        def your_metric_wrapper(y_true, y_pred):
            return your_metric(y_true, y_pred, **hard_coded_kwargs)


"""








