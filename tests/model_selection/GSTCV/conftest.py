# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

import pytest

import re



@pytest.fixture(scope='session')
def standard_cv_int():
    return 4


@pytest.fixture(scope='session')
def standard_error_score():
    return 'raise'




# exc matches ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

@pytest.fixture(scope='session')
def generic_no_attribute_1():
    def foo(_gscv_type, _attr):
        return f"'{_gscv_type}' object has no attribute '{_attr}'"

    return foo


@pytest.fixture(scope='session')
def generic_no_attribute_2():
    def foo(_gscv_type, _attr):
        return f"This '{_gscv_type}' has no attribute '{_attr}'"

    return foo


@pytest.fixture(scope='session')
def generic_no_attribute_3():
    def foo(_gscv_type, _attr):
        return f"{_gscv_type} object has no {_attr} attribute."

    return foo


@pytest.fixture(scope='session')
def _no_refit():
    def foo(_object, _apostrophes: bool, _method):
        if _apostrophes:
            __ = "`refit=False`"
        else:
            __ = "refit=False"

        return (f"This {_object} instance was initialized with {__}. "
            f"{_method} is available only after refitting on the best "
            f"parameters. You can refit an estimator manually using the "
            f"`best_params_` attribute")

    return foo


@pytest.fixture(scope='session')
def _refit_false():

    def foo(_gstcv_type):
        return (f"This {_gstcv_type} instance was initialized "
            f"with `refit=False`. classes_ is available only after "
            "refitting on the best parameters."
        )

    return foo



@pytest.fixture(scope='session')
def _not_fitted():
    def foo(_object):
        return (f"This {_object} instance is not fitted yet.\nCall 'fit' "
            f"with appropriate arguments before using this estimator.")

    return foo


@pytest.fixture(scope='session')
def non_num_X():
    return re.escape(f"dtype='numeric' is not compatible with arrays of "
        f"bytes/strings. Convert your data to numeric values explicitly "
        f"instead."
    )


@pytest.fixture(scope='session')
def partial_feature_names_exc():
    return f"The feature names should match those that were passed during fit."


@pytest.fixture(scope='session')
def multilabel_y():
    return re.escape(f"Classification metrics can't handle a mix of "
        f"multilabel-indicator and binary targets")



@pytest.fixture(scope='session')
def non_binary_y():

    def foo(_gstcv_type):
        return re.escape(f"{_gstcv_type} can only perform thresholding on binary "
            f"targets with values in [0,1]. Pass 'y' as a vector of 0's and 1's.")

    return foo

@pytest.fixture(scope='session')
def different_rows():

    def foo(y_rows, X_rows):
        return re.escape(f"Found input variables with inconsistent "
                         f"numbers of samples: [{y_rows}, {X_rows}]")

    return foo


# END exc matches ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

























