# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


"""

    cv:
        int, iterable, or None, default=None - Sets the cross-validation
        splitting strategy.

        Possible inputs for cv are:
        1) None, to use the default 5-fold cross validation,
        2) integer, must be 2 or greater, to specify the number of folds
            in a (Stratified)KFold,
        3) An iterable yielding pairs of (train, test) split indices as
            arrays.

        For passed iterables:
        This module will convert generators to lists. No validation is
        done beyond verifying that it is an iterable that contains pairs
        of iterables. GSTCV will catch out of range indices and raise
        but any validation beyond that is up to the user outside of
        GSTCV.


"""









