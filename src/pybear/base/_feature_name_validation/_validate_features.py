# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import numpy as np



# pizza 25_01_03_12_53_00 this is taken from _GSTCVMixin



def _validate_features(
    self,
    passed_feature_names: np.ndarray
) -> None:

    # self provides self.feature_names_in_, which is an output of _handle_X_y, gets it on first seen X and stores it

    # passed_feature_names is an output of _handle_X_y, which gets it by X.columns on currently seen data


    """
    Validate that feature names passed to a method via a dataframe
    have the exact names and order as those seen during fit, if fit
    was done with a dataframe. If not, raise ValueError. Return None
    if fit was done with an array or if the object passed to a
    method is an array.


    Parameters
    ----------
    passed_feature_names:
        NDArray[str] - shape (n_features, ), the column header from
        a dataframe passed to a method.


    Return
    ------
    -
        None



    """

    # pizza, this will probably be replaced by a module
    # from pybear.base.  whenever that's done.
    # reconcile with sklearn.BaseEstimator._check_feature_names,
    # MinCountTransformer._val_feature_names(), and whatever crack
    # module pybear comes up with.

    if passed_feature_names is None:
        return

    if not hasattr(self, 'feature_names_in_'):
        return

    pfn = passed_feature_names
    fni = self.feature_names_in_

    if np.array_equiv(pfn, fni):
        return
    else:

        base_err_msg = (f"The feature names should match those that "
                        f"were passed during fit. ")

        if len(pfn) == len(fni) and np.array_equiv(sorted(pfn), sorted(fni)):

            # when out of order
            # ValueError: base_err_msg
            # Feature names must be in the same order as they were in fit.
            addon = (f"\nFeature names must be in the same order as they "
                     f"were in fit.")

        else:

            addon = ''

            UNSEEN = []
            for col in pfn:
                if col not in fni:
                    UNSEEN.append(col)
            if len(UNSEEN) > 0:
                addon += f"\nFeature names unseen at fit time:"
                for col in UNSEEN:
                    addon += f"\n- {col}"
            del UNSEEN

            SEEN = []
            for col in fni:
                if col not in pfn:
                    SEEN.append(col)
            if len(SEEN) > 0:
                addon += f"\nFeature names seen at fit time, yet now missing:"
                for col in SEEN:
                    addon += f"\n- {col}"
            del SEEN

        message = base_err_msg + addon

        raise ValueError(message)












