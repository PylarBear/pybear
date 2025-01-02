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










# # this is called by OneToOneFeatureMixin.get_feature_names_out()
# def _check_feature_names_in(estimator, input_features=None, *, generate_names=True):
#     """Check `input_features` and generate names if needed.
#
#     Commonly used in :term:`get_feature_names_out`.
#
#     Parameters
#     ----------
#     input_features : array-like of str or None, default=None
#         Input features.
#
#         - If `input_features` is `None`, then `feature_names_in_` is
#           used as feature names in. If `feature_names_in_` is not defined,
#           then the following input feature names are generated:
#           `["x0", "x1", ..., "x(n_features_in_ - 1)"]`.
#         - If `input_features` is an array-like, then `input_features` must
#           match `feature_names_in_` if `feature_names_in_` is defined.
#
#     generate_names : bool, default=True
#         Whether to generate names when `input_features` is `None` and
#         `estimator.feature_names_in_` is not defined. This is useful for transformers
#         that validates `input_features` but do not require them in
#         :term:`get_feature_names_out` e.g. `PCA`.
#
#     Returns
#     -------
#     feature_names_in : ndarray of str or `None`
#         Feature names in.
#     """
#
#     feature_names_in_ = getattr(estimator, "feature_names_in_", None)
#     n_features_in_ = getattr(estimator, "n_features_in_", None)
#
#     if input_features is not None:
#         input_features = np.asarray(input_features, dtype=object)
#         if feature_names_in_ is not None and not np.array_equal(
#             feature_names_in_, input_features
#         ):
#             raise ValueError("input_features is not equal to feature_names_in_")
#
#         if n_features_in_ is not None and len(input_features) != n_features_in_:
#             raise ValueError(
#                 "input_features should have length equal to number of "
#                 f"features ({n_features_in_}), got {len(input_features)}"
#             )
#         return input_features
#
#     if feature_names_in_ is not None:
#         return feature_names_in_
#
#     if not generate_names:
#         return
#
#     # Generates feature names if `n_features_in_` is defined
#     if n_features_in_ is None:
#         raise ValueError("Unable to generate feature names without n_features_in_")
#
#     return np.asarray([f"x{i}" for i in range(n_features_in_)], dtype=object)





#
# # sklearn.utils.validation
# # called by _check_feature_names method of BaseEstimator
# def _get_feature_names(X):
#     """Get feature names from X.
#
#     Support for other array containers should place its implementation here.
#
#     Parameters
#     ----------
#     X : {ndarray, dataframe} of shape (n_samples, n_features)
#         Array container to extract feature names.
#
#         - pandas dataframe : The columns will be considered to be feature
#           names. If the dataframe contains non-string feature names, `None` is
#           returned.
#         - All other array containers will return `None`.
#
#     Returns
#     -------
#     names: ndarray or None
#         Feature names of `X`. Unrecognized array containers will return `None`.
#     """
#
#
#
#
#     feature_names = None
#
#     # extract feature names for support array containers
#     if _is_pandas_df(X):
#         # Make sure we can inspect columns names from pandas, even with
#         # versions too old to expose a working implementation of
#         # __dataframe__.column_names() and avoid introducing any
#         # additional copy.
#         # TODO: remove the pandas-specific branch once the minimum supported
#         # version of pandas has a working implementation of
#         # __dataframe__.column_names() that is guaranteed to not introduce any
#         # additional copy of the data without having to impose allow_copy=False
#         # that could fail with other libraries. Note: in the longer term, we
#         # could decide to instead rely on the __dataframe_namespace__ API once
#         # adopted by our minimally supported pandas version.
#         feature_names = np.asarray(X.columns, dtype=object)
#     elif hasattr(X, "__dataframe__"):
#         df_protocol = X.__dataframe__()
#         feature_names = np.asarray(list(df_protocol.column_names()), dtype=object)
#
#     if feature_names is None or len(feature_names) == 0:
#         return
#
#     types = sorted(t.__qualname__ for t in set(type(v) for v in feature_names))
#
#     # mixed type of string and non-string is not supported
#     if len(types) > 1 and "str" in types:
#         raise TypeError(
#             "Feature names are only supported if all input features have string names, "
#             f"but your input has {types} as feature name / column name types. "
#             "If you want feature names to be stored and validated, you must convert "
#             "them all to strings, by using X.columns = X.columns.astype(str) for "
#             "example. Otherwise you can remove feature / column names from your input "
#             "data, or convert them all to a non-string data type."
#         )
#
#     # Only feature names of all strings are supported
#     if len(types) == 1 and types[0] == "str":
#         return feature_names
#
#
#
#
# # this is a method of BaseEstimator which is called by the _validate_data method of BaseEstimator
# def _check_feature_names(self, X, *, reset):
#     """Set or check the `feature_names_in_` attribute.
#
#     .. versionadded:: 1.0
#
#     Parameters
#     ----------
#     X : {ndarray, dataframe} of shape (n_samples, n_features)
#         The input samples.
#
#     reset : bool
#         Whether to reset the `feature_names_in_` attribute.
#         If False, the input will be checked for consistency with
#         feature names of data provided when reset was last True.
#         .. note::
#            It is recommended to call `reset=True` in `fit` and in the first
#            call to `partial_fit`. All other methods that validate `X`
#            should set `reset=False`.
#     """
#
#     if reset:
#         feature_names_in = _get_feature_names(X)
#         if feature_names_in is not None:
#             self.feature_names_in_ = feature_names_in
#         elif hasattr(self, "feature_names_in_"):
#             # Delete the attribute when the estimator is fitted on a new dataset
#             # that has no feature names.
#             delattr(self, "feature_names_in_")
#         return
#
#     fitted_feature_names = getattr(self, "feature_names_in_", None)
#     X_feature_names = _get_feature_names(X)
#
#     if fitted_feature_names is None and X_feature_names is None:
#         # no feature names seen in fit and in X
#         return
#
#     if X_feature_names is not None and fitted_feature_names is None:
#         warnings.warn(
#             f"X has feature names, but {self.__class__.__name__} was fitted without"
#             " feature names"
#         )
#         return
#
#     if X_feature_names is None and fitted_feature_names is not None:
#         warnings.warn(
#             "X does not have valid feature names, but"
#             f" {self.__class__.__name__} was fitted with feature names"
#         )
#         return
#
#     # validate the feature names against the `feature_names_in_` attribute
#     if len(fitted_feature_names) != len(X_feature_names) or np.any(
#         fitted_feature_names != X_feature_names
#     ):
#         message = (
#             "The feature names should match those that were passed during fit.\n"
#         )
#         fitted_feature_names_set = set(fitted_feature_names)
#         X_feature_names_set = set(X_feature_names)
#
#         unexpected_names = sorted(X_feature_names_set - fitted_feature_names_set)
#         missing_names = sorted(fitted_feature_names_set - X_feature_names_set)
#
#         def add_names(names):
#             output = ""
#             max_n_names = 5
#             for i, name in enumerate(names):
#                 if i >= max_n_names:
#                     output += "- ...\n"
#                     break
#                 output += f"- {name}\n"
#             return output
#
#         if unexpected_names:
#             message += "Feature names unseen at fit time:\n"
#             message += add_names(unexpected_names)
#
#         if missing_names:
#             message += "Feature names seen at fit time, yet now missing:\n"
#             message += add_names(missing_names)
#
#         if not missing_names and not unexpected_names:
#             message += (
#                 "Feature names must be in the same order as they were in fit.\n"
#             )
#
#         raise ValueError(message)



