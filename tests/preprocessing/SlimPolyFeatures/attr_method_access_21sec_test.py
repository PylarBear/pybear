# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import sys

import numpy as np
import pandas as pd
import scipy.sparse as ss

from pybear.base import is_fitted
from pybear.base.exceptions import NotFittedError

from pybear.preprocessing import SlimPolyFeatures as SlimPoly

import pytest


bypass = False



# v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^
# FIXTURES

@pytest.fixture(scope='module')
def _shape():
    return (20, 10)


@pytest.fixture(scope='module')
def _dupl(_shape):
    return [[3, 5, _shape[1]-1]]


@pytest.fixture(scope='function')
def _kwargs():
    return {
        'degree': 2,
        'min_degree': 1,
        'scan_X': False,
        'keep': 'first',
        'interaction_only': False,
        'sparse_output': False,
        'feature_name_combiner': "as_indices",
        'equal_nan': True,
        'rtol': 1e-5,
        'atol': 1e-8,
        'n_jobs': 1  # leave this at 1 because of confliction
    }


@pytest.fixture(scope='module')
def _X_np(_X_factory, _dupl, _shape):
    return _X_factory(
        _dupl=_dupl,
        _has_nan=False,
        _dtype='flt',
        _shape=_shape
    )


@pytest.fixture(scope='module')
def _columns(_master_columns, _shape):
    return _master_columns.copy()[:_shape[1]]


@pytest.fixture(scope='module')
def _X_pd(_X_np, _columns):
    return pd.DataFrame(
        data=_X_np,
        columns=_columns
)


@pytest.fixture(scope='module')
def _y_np(_shape):
    return np.random.randint(0, 2, _shape[0])

# END fixtures
# v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^



# ACCESS ATTR BEFORE AND AFTER FIT AND TRANSFORM
@pytest.mark.skipif(bypass is True, reason=f"bypass")
class TestAttrAccessBeforeAndAfterFitAndTransform:


    @staticmethod
    @pytest.fixture
    def _attrs():
        return [
            'n_features_in_',
            'feature_names_in_',
            'poly_combinations_',
            'poly_duplicates_',
            'dropped_poly_duplicates_',
            'kept_poly_duplicates_',
            'poly_constants_'
        ]


    @pytest.mark.parametrize('x_format', ('np', 'pd', 'csc', 'csr', 'coo'))
    def test_attr_access(
        self, _X_np, _X_pd, _y_np, _columns, _kwargs, _shape, _attrs, x_format
    ):

        if x_format == 'np':
            NEW_X = _X_np.copy()
            NEW_Y = np.random.randint(0, 2, _shape[0])
        elif x_format == 'pd':
            NEW_X = _X_pd
            NEW_Y = pd.DataFrame(
                data=np.random.randint(0, 2, _shape[0]), columns=['y']
            )
        elif x_format == 'csc':
            NEW_X = ss.csc_array(_X_np.copy())
            NEW_Y = np.random.randint(0, 2, _shape[0])
        elif x_format == 'csr':
            NEW_X = ss.csr_array(_X_np.copy())
            NEW_Y = np.random.randint(0, 2, _shape[0])
        elif x_format == 'coo':
            NEW_X = ss.coo_array(_X_np.copy())
            NEW_Y = np.random.randint(0, 2, _shape[0])
        else:
            raise Exception


        TestCls = SlimPoly(**_kwargs)

        # BEFORE FIT ***************************************************

        # ALL OF THESE SHOULD GIVE AttributeError/NotFittedError
        # SPF external attrs are @property and raise NotFittedError
        # which is child of AttrError
        # n_features_in_ & feature_names_in_ dont exist before fit
        for attr in _attrs:
            if attr in ['n_features_in_', 'feature_names_in_']:
                with pytest.raises(AttributeError):
                    getattr(TestCls, attr)
            else:
                with pytest.raises(NotFittedError):
                    getattr(TestCls, attr)

        # END BEFORE FIT ***********************************************

        # AFTER FIT ****************************************************

        TestCls.fit(NEW_X, NEW_Y)

        # all attrs should be accessible after fit, the only exception
        # should be feature_names_in_ if not pd
        for attr in _attrs:
            try:
                out = getattr(TestCls, attr)
                if attr == 'feature_names_in_':
                    if x_format == 'pd':
                        assert np.array_equiv(out, _columns), \
                            f"{attr} after fit() != originally passed columns"
                    else:
                        raise AssertionError(
                            f"{x_format} allowed access to 'feature_names_in_"
                        )
                elif attr == 'n_features_in_':
                    assert out == _shape[1]
                else:
                    # not validating accuracy of other module specific outputs
                    pass

            except Exception as e:
                if attr == 'feature_names_in_' and x_format != 'pd':
                    assert isinstance(e, AttributeError)
                else:
                    raise AssertionError(
                        f"unexpected exception {sys.exc_info()[0]} accessing "
                        f"{attr} after fit, x_format == {x_format}"
                    )

        # END AFTER FIT ************************************************

        # AFTER TRANSFORM **********************************************

        TestCls.transform(NEW_X)

        # after transform, should be the exact same condition as after
        # fit, and pass the same tests
        for attr in _attrs:
            try:
                out = getattr(TestCls, attr)
                if attr == 'feature_names_in_':
                    if x_format == 'pd':
                        assert np.array_equiv(out, _columns), \
                            f"{attr} after fit() != originally passed columns"
                    else:
                        raise AssertionError(
                            f"{x_format} allowed access to 'feature_names_in_"
                        )
                elif attr == 'n_features_in_':
                    assert out == _shape[1]
                else:
                    # not validating accuracy of other module specific outputs
                    pass

            except Exception as e:
                if attr == 'feature_names_in_' and x_format != 'pd':
                    assert isinstance(e, AttributeError)
                else:
                    raise AssertionError(
                        f"unexpected exception {sys.exc_info()[0]} accessing "
                        f"{attr} after fit, x_format == {x_format}"
                    )

        # END AFTER TRANSFORM ******************************************

        del NEW_X, NEW_Y, TestCls

# END ACCESS ATTR BEFORE AND AFTER FIT AND TRANSFORM


# ACCESS METHODS BEFORE AND AFTER FIT AND TRANSFORM ***
@pytest.mark.skipif(bypass is True, reason=f"bypass")
class TestMethodAccessBeforeAndAfterFitAndAfterTransform:


    @staticmethod
    def _methods():
        return [
            'fit',
            'fit_transform',
            'get_feature_names_out',
            'get_metadata_routing',
            'get_params',
            'partial_fit',
            'reset',
            'score',
            'set_output',
            'set_params',
            'transform'
        ]


    def test_access_methods_before_fit(self, _X_np, _y_np, _kwargs):

        TestCls = SlimPoly(**_kwargs)

        # **************************************************************
        # vvv BEFORE FIT vvv *******************************************

        # fit()
        assert isinstance(TestCls.fit(_X_np, _y_np), SlimPoly)

        # HERE IS A CONVENIENT PLACE TO TEST reset() ^v^v^v^v^v^v^v^v^v^v^v^v
        # Reset Changes is_fitted To False:
        # fit an instance  (done above)
        # assert the instance is fitted
        assert is_fitted(TestCls) is True
        # call :method: reset
        TestCls.reset()
        # assert the instance is not fitted
        assert is_fitted(TestCls) is False
        # END HERE IS A CONVENIENT PLACE TO TEST reset() ^v^v^v^v^v^v^v^v^v^v^

        # fit_transform()
        assert isinstance(TestCls.fit_transform(_X_np, _y_np), np.ndarray)

        TestCls.reset()

        # get_feature_names_out()
        with pytest.raises(NotFittedError):
            TestCls.get_feature_names_out(None)

        # get_metadata_routing()
        with pytest.raises(NotImplementedError):
            TestCls.get_metadata_routing()

        # get_params()
        assert isinstance(TestCls.get_params(True), dict)

        # inverse_transform()
        # SPF should never have inverse_transform method
        with pytest.raises(AttributeError):
            getattr(TestCls, 'inverse_transform')

        # partial_fit()
        assert isinstance(TestCls.partial_fit(_X_np, _y_np), SlimPoly)

        # reset()
        assert isinstance(TestCls.reset(), SlimPoly)

        # score()
        with pytest.raises(NotFittedError):
            TestCls.score(_X_np, _y_np)

        # set_output()
        assert isinstance(TestCls.set_output(transform='pandas'), SlimPoly)

        # set_params()
        assert isinstance(TestCls.set_params(keep='last'), SlimPoly)
        assert TestCls.keep == 'last'

        # transform()
        with pytest.raises(NotFittedError):
            TestCls.transform(_X_np)

        # END ^^^ BEFORE FIT ^^^ ***************************************
        # **************************************************************


    def test_access_methods_after_fit(
        self, _X_np, _y_np, _columns, _kwargs, _shape
    ):

        # **************************************************************
        # vvv AFTER FIT vvv ********************************************

        TestCls = SlimPoly(**_kwargs)
        TestCls.fit(_X_np, _y_np)

        # fit_transform()
        assert isinstance(TestCls.fit_transform(_X_np), np.ndarray)

        TestCls.reset()

        # fit()
        assert isinstance(TestCls.fit(_X_np), SlimPoly)

        # get_feature_names_out()
        assert isinstance(TestCls.get_feature_names_out(None), np.ndarray)

        # get_metadata_routing()
        with pytest.raises(NotImplementedError):
            TestCls.get_metadata_routing()

        # get_params()
        assert isinstance(TestCls.get_params(True), dict)

        # inverse_transform()
        # SPF should never have inverse_transform
        with pytest.raises(AttributeError):
            getattr(TestCls, 'inverse_transform')

        # partial_fit()
        assert isinstance(TestCls.partial_fit(_X_np), SlimPoly)

        # reset()
        assert isinstance(TestCls.reset(), SlimPoly)

        TestCls.fit(_X_np)

        # score()
        assert TestCls.score(_X_np) is None

        # set_output()
        assert isinstance(TestCls.set_output(transform='default'), SlimPoly)

        # set_params()
        assert isinstance(TestCls.set_params(keep='random'), SlimPoly)

        # transform()
        assert isinstance(TestCls.transform(_X_np), np.ndarray)

        del TestCls

        # END ^^^ AFTER FIT ^^^ ****************************************
        # **************************************************************


    def test_access_methods_after_transform(
        self, _X_np, _y_np, _columns, _kwargs, _shape
    ):

        # **************************************************************
        # vvv AFTER TRANSFORM vvv **************************************
        FittedTestCls = SlimPoly(**_kwargs).fit(_X_np, _y_np)
        TransformedTestCls = SlimPoly(**_kwargs).fit(_X_np, _y_np)
        TransformedTestCls.transform(_X_np)

        # fit_transform()
        assert isinstance(TransformedTestCls.fit_transform(_X_np), np.ndarray)

        # fit()
        assert isinstance(TransformedTestCls.fit(_X_np), SlimPoly)

        TransformedTestCls.transform(_X_np)

        # get_feature_names_out()
        assert isinstance(
            TransformedTestCls.get_feature_names_out(None),
            np.ndarray
        )

        # get_metadata_routing()
        with pytest.raises(NotImplementedError):
            TransformedTestCls.get_metadata_routing()

        # get_params()
        assert TransformedTestCls.get_params(True) == \
                FittedTestCls.get_params(True), \
            f"get_params() after transform() != before transform()"

        # inverse_transform()
        # SPF should never have inverse_transform
        with pytest.raises(AttributeError):
            getattr(TransformedTestCls, 'inverse_transform')

        # partial_fit()
        assert isinstance(TransformedTestCls.partial_fit(_X_np), SlimPoly)
        TransformedTestCls.transform(_X_np)

        # reset()
        assert isinstance(TransformedTestCls.reset(), SlimPoly)
        TransformedTestCls.fit_transform(_X_np)

        # set_output()
        assert isinstance(
            TransformedTestCls.set_output(transform='default'), SlimPoly
        )

        # set_params()
        assert isinstance(TransformedTestCls.set_params(keep='first'), SlimPoly)

        # transform()
        assert isinstance(TransformedTestCls.fit_transform(_X_np), np.ndarray)

        del FittedTestCls, TransformedTestCls

        # END ^^^ AFTER TRANSFORM ^^^ **********************************
        # **************************************************************

# END ACCESS METHODS BEFORE AND AFTER FIT AND TRANSFORM













