# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pytest

from pybear.preprocessing import InterceptManager as IM

import sys
import numpy as np
import pandas as pd
import scipy.sparse as ss

from pybear.exceptions import NotFittedError






bypass = False



# v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^
# FIXTURES

@pytest.fixture(scope='module')
def _shape():
    return (20, 10)


@pytest.fixture(scope='function')
def _kwargs():
    return {
        'keep': 'last',
        'rtol': 1e-5,
        'atol': 1e-8,
        'equal_nan': False,
        'n_jobs': 1  # leave this at 1 because of confliction
    }


@pytest.fixture(scope='module')
def _const(_shape):
    return {3:0, 5:1, _shape[1]-1:2}


@pytest.fixture(scope='module')
def _dum_X(_X_factory, _const, _shape):
    return _X_factory(
        _constants=_const,
        _has_nan=False,
        _dtype='flt',
        _shape=_shape
    )


@pytest.fixture(scope='module')
def _columns(_master_columns, _shape):
    return _master_columns.copy()[:_shape[1]]


@pytest.fixture(scope='module')
def _X_pd(_dum_X, _columns):
    return pd.DataFrame(
        data=_dum_X,
        columns=_columns
)


# END fixtures
# v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^



# ACCESS ATTR BEFORE AND AFTER FIT AND TRANSFORM
@pytest.mark.skipif(bypass is True, reason=f"bypass")
class TestAttrAccessAndAccuracyBeforeAndAfterFitAndTransform:


    @staticmethod
    @pytest.fixture
    def _attrs():
        return [
            'n_features_in_',
            'feature_names_in_',
            'constant_columns_',
            'kept_columns_',
            'removed_columns_',
            'column_mask_'
        ]


    @pytest.mark.parametrize('x_format', ('np', 'pd', 'csc', 'csr', 'coo'))
    def test_attr_accuracy(
        self, _dum_X, _X_pd, _columns, _kwargs, _shape, _attrs, x_format
    ):

        if x_format == 'np':
            NEW_X = _dum_X.copy()
            NEW_Y = np.random.randint(0, 2, _shape[0])
        elif x_format == 'pd':
            NEW_X = _X_pd
            NEW_Y = pd.DataFrame(
                data=np.random.randint(0, 2, _shape[0]), columns=['y']
            )
        elif x_format == 'csc':
            NEW_X = ss.csc_array(_dum_X.copy())
            NEW_Y = np.random.randint(0, 2, _shape[0])
        elif x_format == 'csr':
            NEW_X = ss.csr_array(_dum_X.copy())
            NEW_Y = np.random.randint(0, 2, _shape[0])
        elif x_format == 'coo':
            NEW_X = ss.coo_array(_dum_X.copy())
            NEW_Y = np.random.randint(0, 2, _shape[0])
        else:
            raise Exception


        TestCls = IM(**_kwargs)

        # BEFORE FIT ***************************************************

        # ALL OF THESE SHOULD GIVE AttributeError
        for attr in _attrs:
            with pytest.raises(AttributeError):
                getattr(TestCls, attr)

        # END BEFORE FIT ***********************************************

        # AFTER FIT ****************************************************

        TestCls.fit(NEW_X, NEW_Y)

        # all attrs should be accessible after fit, the only exception
        # should be feature_names_in_ if numpy
        for attr in _attrs:
            try:
                out = getattr(TestCls, attr)
                if attr == 'feature_names_in_' and x_format == 'pd':
                    assert np.array_equiv(out, _columns), \
                        f"{attr} after fit() != originally passed columns"
                elif attr == 'n_features_in_':
                    assert out == _shape[1], \
                        f"{attr} after fit() != number of originally passed columns"
            except:
                if attr == 'feature_names_in_' and x_format != 'pd':
                    assert isinstance(sys.exc_info()[0](), AttributeError)
                else:
                    raise

        # END AFTER FIT ************************************************

        # AFTER TRANSFORM **********************************************

        # after transform, should be the exact same condition as after
        # fit, and pass the same tests
        for attr in _attrs:
            try:
                out = getattr(TestCls, attr)
                if attr == 'feature_names_in_' and x_format == 'pd':
                    assert np.array_equiv(out, _columns), \
                        f"{attr} after fit() != originally passed columns"
                elif attr == 'n_features_in_':
                    assert out == _shape[1], \
                        f"{attr} after fit() != number of originally passed columns"
            except:
                if attr == 'feature_names_in_' and x_format != 'pd':
                    assert isinstance(sys.exc_info()[1], AttributeError)
                else:
                    raise AssertionError(
                        f"unexpected exception accessing {attr} after fit, "
                        f"x_format == {x_format}"
                    )

        # END AFTER TRANSFORM ******************************************

        del NEW_X, NEW_Y, TestCls

# END ACCESS ATTR BEFORE AND AFTER FIT AND TRANSFORM


# ACCESS METHODS BEFORE AND AFTER FIT AND TRANSFORM ***
@pytest.mark.skipif(bypass is True, reason=f"bypass")
class TestMethodAccessAndAccuracyBeforeAndAfterFitAndAfterTransform:


    @staticmethod
    def _methods():
        return [
            'fit',
            'fit_transform',
            'get_feature_names_out',
            'get_metadata_routing',
            'get_params',
            'inverse_transform',
            'partial_fit',
            'set_output',
            'set_params',
            'transform'
        ]


    def test_access_methods_before_fit(self, _dum_X, _X_pd, _kwargs):

        TestCls = IM(**_kwargs)

        # **************************************************************
        # vvv BEFORE FIT vvv *******************************************

        # fit()
        # fit_transform()

        # get_feature_names_out()
        with pytest.raises(NotFittedError):
            TestCls.get_feature_names_out(None)

        # get_metadata_routing()
        with pytest.raises(NotImplementedError):
            TestCls.get_metadata_routing()

        # get_params()
        TestCls.get_params(True)

        # inverse_transform()
        with pytest.raises(NotFittedError):
            TestCls.inverse_transform(_dum_X)

        # partial_fit()

        # set_output()
        TestCls.set_output(transform='pandas')

        # set_params()
        # KEYS = [
        #     'keep': 'first',
        #     'rtol': 1e-5,
        #     'atol': 1e-8,
        #     'equal_nan': False,
        #     'n_jobs': -1
        # ]
        TestCls.set_params(keep='last')


        # transform()
        with pytest.raises(NotFittedError):
            TestCls.transform(_dum_X)


        # END ^^^ BEFORE FIT ^^^ ***************************************
        # **************************************************************


    def test_access_methods_after_fit(
        self, _dum_X, _X_pd, _columns, _kwargs, _shape
    ):

        y = np.random.randint(0,2,_shape[0])

        # **************************************************************
        # vvv AFTER FIT vvv ********************************************

        TestCls = IM(**_kwargs)
        TestCls.fit(_dum_X, y)

        # fit()
        # fit_transform()

        # get_feature_names_out()
        assert isinstance(TestCls.get_feature_names_out(None), np.ndarray)

        # get_metadata_routing()
        with pytest.raises(NotImplementedError):
            TestCls.get_metadata_routing()

        # get_params()
        TestCls.get_params(True)

        del TestCls

        # inverse_transform() ********************
        TestCls = IM(**_kwargs)
        TestCls.fit(_dum_X, y)  # X IS NP ARRAY

        # VALIDATION OF X GOING INTO inverse_transform IS HANDLED BY
        # sklearn check_array, LET IT RAISE WHATEVER
        for junk_x in [[], [[]], None, 'junk_string', 3, np.pi]:
            with pytest.raises(Exception):
                TestCls.inverse_transform(junk_x)

        # SHOULD RAISE ValueError WHEN COLUMNS DO NOT EQUAL NUMBER OF
        # RETAINED COLUMNS
        TRFM_X = TestCls.transform(_dum_X)
        TRFM_MASK = TestCls.column_mask_
        __ = np.array(_columns)
        for obj_type in ['np', 'pd']:
            for diff_cols in ['more', 'less', 'same']:
                if diff_cols == 'same':
                    TEST_X = TRFM_X.copy()
                    if obj_type == 'pd':
                        TEST_X = pd.DataFrame(
                            data=TEST_X, columns=__[TRFM_MASK]
                        )
                elif diff_cols == 'less':
                    TEST_X = TRFM_X[:, :2].copy()
                    if obj_type == 'pd':
                        TEST_X = pd.DataFrame(
                            data=TEST_X, columns=__[TRFM_MASK][:2]
                        )
                elif diff_cols == 'more':
                    TEST_X = np.hstack((TRFM_X.copy(), TRFM_X.copy()))
                    if obj_type == 'pd':
                        _COLUMNS = np.hstack((
                            __[TRFM_MASK],
                            np.char.upper(__[TRFM_MASK])
                        ))
                        TEST_X = pd.DataFrame(data=TEST_X, columns=_COLUMNS)

                if diff_cols == 'same':
                    TestCls.inverse_transform(TEST_X)
                else:
                    with pytest.raises(ValueError):
                        TestCls.inverse_transform(TEST_X)

        INV_TRFM_X = TestCls.inverse_transform(TRFM_X)
        if isinstance(TRFM_X, np.ndarray):
            assert INV_TRFM_X.flags['C_CONTIGUOUS'] is True

        assert isinstance(INV_TRFM_X, np.ndarray), \
            f"output of inverse_transform() is not a numpy array"
        assert INV_TRFM_X.shape[0] == TRFM_X.shape[0], \
            f"rows in output of inverse_transform() do not match input rows"
        assert INV_TRFM_X.shape[1] == TestCls.n_features_in_, \
            (f"columns in output of inverse_transform() do not match "
             f"originally fitted columns")

        assert np.array_equiv( INV_TRFM_X, _dum_X), \
            f"inverse transform of transformed data does not equal original data"

        assert np.array_equiv(
            TRFM_X.astype(str),
            INV_TRFM_X[:, TestCls.column_mask_].astype(str)
        ), (f"output of inverse_transform() does not reduce back to the output "
            f"of transform()")

        del junk_x, TRFM_X, TRFM_MASK, obj_type, diff_cols
        del TEST_X, INV_TRFM_X, TestCls

        # END inverse_transform() **********

        TestCls = IM(**_kwargs)

        # partial_fit()
        # ** _reset()

        # set_output()
        TestCls.set_output(transform='pandas')

        # set_params()
        TestCls.set_params(keep='random')

        del TestCls

        # transform()

        # END ^^^ AFTER FIT ^^^ ****************************************
        # **************************************************************


    def test_access_methods_after_transform(
        self, _dum_X, _columns, _kwargs, _shape
    ):

        y = np.random.randint(0, 2, _shape[0])

        # **************************************************************
        # vvv AFTER TRANSFORM vvv **************************************
        FittedTestCls = IM(**_kwargs).fit(_dum_X, y)
        TransformedTestCls = IM(**_kwargs).fit(_dum_X, y)
        TRFM_X = TransformedTestCls.transform(_dum_X)

        # fit()
        # fit_transform()

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

        # inverse_transform() ************

        assert np.array_equiv(
            FittedTestCls.inverse_transform(TRFM_X).astype(str),
            TransformedTestCls.inverse_transform(TRFM_X).astype(str)), \
            (f"inverse_transform(TRFM_X) after transform() != "
             f"inverse_transform(TRFM_X) before transform()")

        # END inverse_transform() **********

        # partial_fit()
        # ** _reset()

        # set_output()
        TransformedTestCls.set_output(transform='pandas')
        TransformedTestCls.transform(_dum_X)

        del TransformedTestCls

        # set_params()
        TestCls = IM(**_kwargs)
        TestCls.set_params(keep='first')

        # transform()

        del FittedTestCls, TestCls, TRFM_X

        # END ^^^ AFTER TRANSFORM ^^^ **********************************
        # **************************************************************

# END ACCESS METHODS BEFORE AND AFTER FIT AND TRANSFORM













