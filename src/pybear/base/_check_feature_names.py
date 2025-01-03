# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



# pizza


# def check_feature_names_in(
#
# )
#
#     """
#
#
#     """














# this is a method of BaseEstimator which is called by the _validate_data method of BaseEstimator
def check_feature_names(self, X, *, reset):

    # pizza dont delete this, compare it against what is in _GSTCVMixin

    """Set or check the `feature_names_in_` attribute.

    .. versionadded:: 1.0

    Parameters
    ----------
    X : {ndarray, dataframe} of shape (n_samples, n_features)
        The input samples.

    reset : bool
        Whether to reset the `feature_names_in_` attribute.
        If False, the input will be checked for consistency with
        feature names of data provided when reset was last True.
        .. note::
           It is recommended to call `reset=True` in `fit` and in the first
           call to `partial_fit`. All other methods that validate `X`
           should set `reset=False`.
    """

    if reset:
        feature_names_in = _get_feature_names(X)
        if feature_names_in is not None:
            self.feature_names_in_ = feature_names_in
        elif hasattr(self, "feature_names_in_"):
            # Delete the attribute when the estimator is fitted on a new dataset
            # that has no feature names.
            delattr(self, "feature_names_in_")
        return

    fitted_feature_names = getattr(self, "feature_names_in_", None)
    X_feature_names = _get_feature_names(X)

    if fitted_feature_names is None and X_feature_names is None:
        # no feature names seen in fit and in X
        return

    if X_feature_names is not None and fitted_feature_names is None:
        warnings.warn(
            f"X has feature names, but {self.__class__.__name__} was fitted without"
            " feature names"
        )
        return

    if X_feature_names is None and fitted_feature_names is not None:
        warnings.warn(
            "X does not have valid feature names, but"
            f" {self.__class__.__name__} was fitted with feature names"
        )
        return

    # validate the feature names against the `feature_names_in_` attribute
    if len(fitted_feature_names) != len(X_feature_names) or np.any(
        fitted_feature_names != X_feature_names
    ):
        message = (
            "The feature names should match those that were passed during fit.\n"
        )
        fitted_feature_names_set = set(fitted_feature_names)
        X_feature_names_set = set(X_feature_names)

        unexpected_names = sorted(X_feature_names_set - fitted_feature_names_set)
        missing_names = sorted(fitted_feature_names_set - X_feature_names_set)

        def add_names(names):
            output = ""
            max_n_names = 5
            for i, name in enumerate(names):
                if i >= max_n_names:
                    output += "- ...\n"
                    break
                output += f"- {name}\n"
            return output

        if unexpected_names:
            message += "Feature names unseen at fit time:\n"
            message += add_names(unexpected_names)

        if missing_names:
            message += "Feature names seen at fit time, yet now missing:\n"
            message += add_names(missing_names)

        if not missing_names and not unexpected_names:
            message += (
                "Feature names must be in the same order as they were in fit.\n"
            )

        raise ValueError(message)



