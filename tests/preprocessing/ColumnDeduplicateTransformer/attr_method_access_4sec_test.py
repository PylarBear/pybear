# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pytest

from pybear.preprocessing import ColumnDeduplicateTransformer as CDT

import sys
import numpy as np
import pandas as pd

from pybear.base import NotFittedError




bypass = False



# v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^
# FIXTURES

@pytest.fixture(scope='module')
def _shape():
    return (20, 10)


@pytest.fixture(scope='function')
def _kwargs():
    return {
        'keep': 'first',
        'do_not_drop': None,
        'conflict': 'raise',
        'rtol': 1e-5,
        'atol': 1e-8,
        'equal_nan': False,
        'n_jobs': 1   # leave set at 1 because of confliction
    }


@pytest.fixture(scope='module')
def _dupl(_shape):
    return [[3, 5, _shape[1]-1]]


@pytest.fixture(scope='module')
def _dum_X(_X_factory, _dupl, _shape):
    return _X_factory(_dupl=_dupl, _has_nan=False, _dtype='flt', _shape=_shape)


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



# ACCESS ATTR BEFORE AND AFTER FIT AND TRANSFORM, ATTR ACCURACY
@pytest.mark.skipif(bypass is True, reason=f"bypass")
class TestAttrAccessAndAccuracyBeforeAndAfterFitAndTransform:


    @staticmethod
    @pytest.fixture
    def _attrs():
        return [
            'n_features_in_',
            'feature_names_in_',
            'duplicates_',
            'removed_columns_',
            'column_mask_'
        ]


    @pytest.mark.parametrize('x_format', ('np', 'pd'))
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
        else:
            raise Exception


        TestCls = CDT(**_kwargs)

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
        # duplicates_, removed_columns_, & column_mask_ tested elsewhere
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
                if attr == 'feature_names_in_' and x_format == 'np':
                    assert isinstance(sys.exc_info()[0](), AttributeError)
                else:
                    raise

        # END AFTER FIT ************************************************

        # AFTER TRANSFORM **********************************************

        # after transform, should be the exact same condition as after
        # fit, and pass the same tests
        # duplicates_, removed_columns_, & column_mask_ tested elsewhere
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
                if attr == 'feature_names_in_' and x_format == 'np':
                    assert isinstance(sys.exc_info()[1], AttributeError)
                else:
                    raise AssertionError(
                        f"unexpected exception accessing {attr} after fit, "
                        f"x_format == {x_format}"
                    )

        # END AFTER TRANSFORM ******************************************

        del NEW_X, NEW_Y, TestCls

# END ACCESS ATTR BEFORE AND AFTER FIT AND TRANSFORM, ATTR ACCURACY


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


    def test_access_methods_before_fit(self, _dum_X, _kwargs):

        TestCls = CDT(**_kwargs)

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
        #     'do_not_drop': None,
        #     'conflict': 'raise',
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
        self, _dum_X, _columns, _kwargs, _shape
    ):

        y = np.random.randint(0,2,_shape[0])

        # **************************************************************
        # vvv AFTER FIT vvv ********************************************

        TestCls = CDT(**_kwargs)
        TestCls.fit(_dum_X, y)

        # fit()
        # fit_transform()

        # get_feature_names_out() v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^
        # vvv NO COLUMN NAMES PASSED (NP) vvv
        # **** CAN ONLY TAKE LIST-TYPE OF STRS OR None
        JUNK_ARGS = [
            float('inf'), np.pi, 'garbage', {'junk': 3}, [*range(len(_columns))]
        ]

        for junk_arg in JUNK_ARGS:
            with pytest.raises(ValueError):
                TestCls.get_feature_names_out(junk_arg)

        del JUNK_ARGS

        # WITH NO HEADER PASSED AND input_features=None, SHOULD RETURN
        # ['x0', ..., 'x(n-1)][COLUMN MASK]
        _COLUMNS = np.array([f"x{i}" for i in range(len(_columns))])
        assert np.array_equiv(
            TestCls.get_feature_names_out(None),
            _COLUMNS[TestCls.column_mask_]
        ), \
            (f"get_feature_names_out(None) after fit() != sliced array of "
            f"generic headers")

        # WITH NO HEADER PASSED, SHOULD RAISE ValueError IF
        # len(input_features) != n_features_in_
        with pytest.raises(ValueError):
            TestCls.get_feature_names_out(
                [f"x{i}" for i in range(2 * len(_columns))]
            )

        # WHEN NO HEADER PASSED TO (partial_)fit() AND VALID input_features,
        # SHOULD RETURN SLICED PASSED COLUMNS
        RETURNED_FROM_GFNO = TestCls.get_feature_names_out(_columns)
        assert isinstance(RETURNED_FROM_GFNO, np.ndarray), \
            (f"get_feature_names_out should return numpy.ndarray, but "
             f"returned {type(RETURNED_FROM_GFNO)}")

        _ACTIVE_COLUMNS = np.array(_columns)[TestCls.column_mask_]
        assert np.array_equiv(RETURNED_FROM_GFNO, _ACTIVE_COLUMNS), \
            f"get_feature_names_out() did not return original columns"

        del junk_arg, RETURNED_FROM_GFNO, TestCls, _ACTIVE_COLUMNS

        # END ^^^ NO COLUMN NAMES PASSED (NP) ^^^

        # vvv COLUMN NAMES PASSED (PD) vvv

        TestCls = CDT(**_kwargs)
        TestCls.fit(pd.DataFrame(data=_dum_X, columns=_columns), y)

        # WITH HEADER PASSED AND input_features=None, SHOULD RETURN
        # SLICED ORIGINAL COLUMNS
        _ACTIVE_COLUMNS = np.array(_columns)[TestCls.column_mask_]
        assert np.array_equiv(TestCls.get_feature_names_out(None), _ACTIVE_COLUMNS), \
            f"get_feature_names_out(None) after fit() != originally passed columns"
        del _ACTIVE_COLUMNS

        # WITH HEADER PASSED, SHOULD RAISE TypeError IF input_features
        # FOR DISALLOWED TYPES

        JUNK_COL_NAMES = [
            [*range(len(_columns))], [*range(2 * len(_columns))], {'a': 1, 'b': 2}
        ]
        for junk_col_names in JUNK_COL_NAMES:
            with pytest.raises(ValueError):
                TestCls.get_feature_names_out(junk_col_names)

        del JUNK_COL_NAMES

        # WITH HEADER PASSED, SHOULD RAISE ValueError IF input_features DOES
        # NOT EXACTLY MATCH ORIGINALLY FIT COLUMNS
        JUNK_COL_NAMES = [
            np.char.upper(_columns), np.hstack((_columns, _columns)), []
        ]
        for junk_col_names in JUNK_COL_NAMES:
            with pytest.raises(ValueError):
                TestCls.get_feature_names_out(junk_col_names)

        # WHEN HEADER PASSED TO (partial_)fit() AND input_features IS THAT HEADER,
        # SHOULD RETURN SLICED VERSION OF THAT HEADER

        RETURNED_FROM_GFNO = TestCls.get_feature_names_out(_columns)
        assert isinstance(RETURNED_FROM_GFNO, np.ndarray), \
            (f"get_feature_names_out should return numpy.ndarray, "
             f"but returned {type(RETURNED_FROM_GFNO)}")

        assert np.array_equiv(
            RETURNED_FROM_GFNO,
            np.array(_columns)[TestCls.column_mask_]
        ), \
            f"get_feature_names_out() did not return original columns"

        del junk_col_names, RETURNED_FROM_GFNO
        # END ^^^ COLUMN NAMES PASSED (PD) ^^^

        # END get_feature_names_out() v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^

        # get_metadata_routing()
        with pytest.raises(NotImplementedError):
            TestCls.get_metadata_routing()

        # get_params()
        TestCls.get_params(True)

        del TestCls

        # inverse_transform()
        TestCls = CDT(**_kwargs)
        TestCls.fit(_dum_X, y)  # X IS NP ARRAY
        TestCls.inverse_transform(_dum_X[:, TestCls.column_mask_])

        del TestCls

        # partial_fit()
        # ** _reset()

        # set_output()
        TestCls = CDT(**_kwargs)
        TestCls.fit(_dum_X, y)  # X IS NP ARRAY
        TestCls.set_output(transform='pandas')

        # set_params()
        TestCls.set_params(keep='random')

        # transform()
        TestCls.transform(_dum_X)

        # END ^^^ AFTER FIT ^^^ ****************************************
        # **************************************************************


    def test_access_methods_after_transform(
        self, _dum_X, _columns, _kwargs, _shape
    ):

        y = np.random.randint(0, 2, _shape[0])

        # **************************************************************
        # vvv AFTER TRANSFORM vvv **************************************
        FittedTestCls = CDT(**_kwargs).fit(_dum_X, y)
        TransformedTestCls = CDT(**_kwargs).fit(_dum_X, y)
        TRFM_X = TransformedTestCls.transform(_dum_X)

        # fit()
        # fit_transform()

        # get_feature_names_out() **************************************
        # vvv NO COLUMN NAMES PASSED (NP) vvv

        # # WHEN NO HEADER PASSED TO (partial_)fit() AND VALID input_features,
        # SHOULD RETURN ORIGINAL (SLICED) COLUMNS
        RETURNED_FROM_GFNO = TransformedTestCls.get_feature_names_out(_columns)

        _ACTIVE_COLUMNS = np.array(_columns)[TransformedTestCls.column_mask_]
        assert np.array_equiv(RETURNED_FROM_GFNO, _ACTIVE_COLUMNS), \
            (f"get_feature_names_out() after transform did not return "
             f"sliced original columns")

        del RETURNED_FROM_GFNO
        # END ^^^ NO COLUMN NAMES PASSED (NP) ^^^

        # vvv COLUMN NAMES PASSED (PD) vvv
        PDTransformedTestCls = CDT(**_kwargs)
        PDTransformedTestCls.fit_transform(
            pd.DataFrame(data=_dum_X, columns=_columns), y
        )

        # WITH HEADER PASSED AND input_features=None,
        # SHOULD RETURN SLICED ORIGINAL COLUMNS
        assert np.array_equiv(
                PDTransformedTestCls.get_feature_names_out(None),
                np.array(_columns)[PDTransformedTestCls.column_mask_]
        ), (f"get_feature_names_out(None) after transform() != originally "
            f"passed columns")

        del PDTransformedTestCls
        # END ^^^ COLUMN NAMES PASSED (PD) ^^^

        # END get_feature_names_out() **********************************

        # get_metadata_routing()
        with pytest.raises(NotImplementedError):
            TransformedTestCls.get_metadata_routing()

        # get_params()
        assert TransformedTestCls.get_params(True) == \
                FittedTestCls.get_params(True), \
            f"get_params() after transform() != before transform()"

        # inverse_transform()
        assert np.array_equiv(
            FittedTestCls.inverse_transform(TRFM_X).astype(str),
            TransformedTestCls.inverse_transform(TRFM_X).astype(str)
        ), (f"inverse_transform(TRFM_X) after transform() != "
             f"inverse_transform(TRFM_X) before transform()")


        # partial_fit()
        # ** _reset()

        # set_output()
        TransformedTestCls.set_output(transform='pandas')
        TransformedTestCls.transform(_dum_X)


        # set_params()
        TransformedTestCls.set_params(keep='first')

        # transform()
        TransformedTestCls.transform(_dum_X)

        del FittedTestCls, TransformedTestCls, TRFM_X

        # END ^^^ AFTER TRANSFORM ^^^ **********************************
        # **************************************************************

# END ACCESS METHODS BEFORE AND AFTER FIT AND TRANSFORM













