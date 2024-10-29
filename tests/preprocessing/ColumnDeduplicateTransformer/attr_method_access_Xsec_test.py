# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.preprocessing import ColumnDeduplicateTransformer as CDT

import pytest

import numpy as np
import pandas as pd

from sklearn.exceptions import NotFittedError



pytest.skip(reason=f"pizza isnt done", allow_module_level=True)



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
        'n_jobs': -1
    }


@pytest.fixture(scope='module')
def _dum_X(_X_factory, _shape):
    return _X_factory(_dupl=None, _has_nan=False, _dtype='flt', _shape=_shape)


# pizza is this even used
@pytest.fixture(scope='function')
def _std_dupl(_shape):
    return [[0, 4], [3, 5, _shape[1]-1]]


@pytest.fixture(scope='module')
def _columns(_master_columns, _shape):
    return _master_columns.copy()[:_shape[1]]


@pytest.fixture(scope='function')
def _bad_columns(_master_columns, _shape):
    return _master_columns.copy()[-_shape[1]:]


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
class TestAttrAccuracyBeforeAndAfterFitAndTransform:

    @staticmethod
    def _attrs():
        return [
            'n_features_in_',
            'feature_names_in_',
            'duplicates_',
            'removed_columns_',
            'column_mask_'
        ]


    def test_attr_accuracy(
        self, _dum_X, _columns, _kwargs, _shape, _attrs
    ):

        NEW_X = _dum_X.copy()
        NEW_X_DF = pd.DataFrame(data=_dum_X, columns=_columns)

        NEW_Y = np.random.randint(0,2,_shape[0])
        NEW_Y_DF = pd.DataFrame(data=NEW_Y, _columns=['y'])

        # BEFORE FIT ***************************************************
        TestCls = CDT(**_kwargs)

        # ALL OF THESE SHOULD GIVE AttributeError
        for attr in _attrs:
            with pytest.raises(AttributeError):
                getattr(TestCls, attr)

        del TestCls
        # END BEFORE FIT ***********************************************

        # AFTER FIT ****************************************************
        for data_dtype in ['np', 'pd']:
            if data_dtype == 'np':
                TEST_X, TEST_Y = NEW_X.copy(), NEW_Y.copy()
            elif data_dtype == 'pd':
                TEST_X, TEST_Y = NEW_X_DF.copy(), NEW_Y_DF.copy()

            TestCls = CDT(**_kwargs)
            TestCls.fit(TEST_X, TEST_Y)

            # ONLY EXCEPTION SHOULD BE feature_names_in_ IF NUMPY
            if data_dtype == 'pd':
                assert np.array_equiv(TestCls.feature_names_in_, _columns), \
                    f"feature_names_in_ after fit() != originally passed columns"
            elif data_dtype == 'np':
                with pytest.raises(AttributeError):
                    TestCls.feature_names_in_

            assert TestCls.n_features_in_ == _shape[0], \
                f"n_features_in_ after fit() != number of originally passed columns"

        del data_dtype, TEST_X, TEST_Y, TestCls

        # END AFTER FIT ************************************************

        # AFTER TRANSFORM **********************************************

        for data_dtype in ['np', 'pd']:

            if data_dtype == 'np':
                TEST_X, TEST_Y = NEW_X.copy(), NEW_Y.copy()
            elif data_dtype == 'pd':
                TEST_X, TEST_Y = NEW_X_DF.copy(), NEW_Y_DF.copy()

            TestCls = CDT(**_kwargs)
            TestCls.fit_transform(TEST_X, TEST_Y)

            # ONLY EXCEPTION SHOULD BE feature_names_in_ WHEN NUMPY
            if data_dtype == 'pd':
                assert np.array_equiv(TestCls.feature_names_in_, _columns), \
                    f"feature_names_in_ after fit() != originally passed columns"
            elif data_dtype == 'np':
                with pytest.raises(AttributeError):
                    TestCls.feature_names_in_

            assert TestCls.n_features_in_ == _shape[1], \
                f"n_features_in_ after fit() != number of originally passed columns"

        del data_dtype, TEST_X, TEST_Y, TestCls
        # END AFTER TRANSFORM ******************************************

        del NEW_X, NEW_Y, NEW_X_DF, NEW_Y_DF

# END ACCESS ATTR BEFORE AND AFTER FIT AND TRANSFORM, ATTR ACCURACY


# ACCESS METHODS BEFORE AND AFTER FIT AND TRANSFORM ***
@pytest.mark.skipif(bypass is True, reason=f"bypass")
class Test1RecursionAccessMethodsBeforeAndAfterFitAndTransform:

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

        # get_row_support()
        with pytest.raises(NotFittedError):
            TestCls.get_row_support(True)

        # get_support()
        with pytest.raises(NotFittedError):
            TestCls.get_support(True)

        # inverse_transform()
        with pytest.raises(NotFittedError):
            TestCls.inverse_transform(_dum_X)

        # partial_fit()
        # ** _reset()

        # set_output()
        TestCls.set_output(transform='pandas_dataframe')

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
        VALUES = [4, False, False, [0], False, [2], True, True, 2, 4]
        test_kwargs = dict((zip(KEYS, VALUES)))
        TestCls.set_params(**test_kwargs)
        ATTRS = [
            TestCls._count_threshold, TestCls._ignore_float_columns,
            TestCls._ignore_non_binary_integer_columns, TestCls._ignore_columns,
            TestCls._ignore_nan, TestCls._handle_as_bool, TestCls._delete_axis_0,
            TestCls._reject_unseen_values, TestCls._max_recursions, TestCls._n_jobs
        ]
        for _key, _attr, _value in zip(KEYS, ATTRS, VALUES):
            assert _attr == _value, f'set_params() did not set {_key}'

        # DEMONSTRATE EXCEPTS FOR UNKNOWN PARAM
        with pytest.raises(ValueError):
            TestCls.set_params(garbage=1)

        del TestCls, KEYS, VALUES, ATTRS

        TestCls = CDT(**_kwargs)
        # transform()
        with pytest.raises(NotFittedError):
            TestCls.transform(_dum_X)


        # END ^^^ BEFORE FIT ^^^ ***************************************
        # **************************************************************

    def test_access_methods_after_fit(self, _X, _columns, _kwargs, _shape):

        X = _X()   # pizza figure out what this needs to be
        y = np.random.randint(0,2,_shape[0])

        # **************************************************************
        # vvv AFTER FIT vvv ********************************************

        TestCls = CDT(**_kwargs)
        TestCls.fit(X, y)

        # ** _base_fit()
        # ** _check_is_fitted()

        # ** test_threshold()
        TestCls.test_threshold()
        print(f'^^^ mask building instructions should be displayed above ^^^')

        # fit()
        # fit_transform()

        # get_feature_names_out() **************************************
        # vvv NO COLUMN NAMES PASSED (NP) vvv
        # **** CAN ONLY TAKE LIST-TYPE OF STRS OR None
        JUNK_ARGS = [float('inf'), np.pi, 'garbage', {'junk': 3},
                     [*range(len(_columns))]
        ]

        for junk_arg in JUNK_ARGS:
            with pytest.raises(TypeError):
                TestCls.get_feature_names_out(junk_arg)

        del JUNK_ARGS

        # WITH NO HEADER PASSED AND input_features=None, SHOULD RETURN
        # ['x0', ..., 'x(n-1)][COLUMN MASK]
        _COLUMNS = np.array([f"x{i}" for i in range(len(_columns))])
        ACTIVE_COLUMNS = _COLUMNS[TestCls.get_support(False)]
        del _COLUMNS
        assert np.array_equiv( TestCls.get_feature_names_out(None), ACTIVE_COLUMNS), \
            (f"get_feature_names_out(None) after fit() != sliced array of "
            f"generic headers")
        del ACTIVE_COLUMNS

        # WITH NO HEADER PASSED, SHOULD RAISE ValueError IF
        # len(input_features) != n_features_in_
        with pytest.raises(ValueError):
            TestCls.get_feature_names_out([f"x{i}" for i in range(2 * len(_columns))])

        # WHEN NO HEADER PASSED TO (partial_)fit() AND VALID input_features,
        # SHOULD RETURN SLICED PASSED COLUMNS
        RETURNED_FROM_GFNO = TestCls.get_feature_names_out(_columns)
        assert isinstance(RETURNED_FROM_GFNO, np.ndarray), \
            (f"get_feature_names_out should return numpy.ndarray, but "
             f"returned {type(RETURNED_FROM_GFNO)}")
        _ACTIVE_COLUMNS = np.array(_columns)[TestCls.get_support(False)]
        assert np.array_equiv(RETURNED_FROM_GFNO, _ACTIVE_COLUMNS), \
            f"get_feature_names_out() did not return original columns"

        del junk_arg, RETURNED_FROM_GFNO, TestCls, _ACTIVE_COLUMNS

        # END ^^^ NO COLUMN NAMES PASSED (NP) ^^^

        # vvv COLUMN NAMES PASSED (PD) vvv

        TestCls = CDT(**_kwargs)
        TestCls.fit(pd.DataFrame(data=X, columns=_columns), y)

        # WITH HEADER PASSED AND input_features=None, SHOULD RETURN
        # SLICED ORIGINAL COLUMNS
        _ACTIVE_COLUMNS = np.array(_columns)[TestCls.get_support(False)]
        assert np.array_equiv(TestCls.get_feature_names_out(None), _ACTIVE_COLUMNS), \
            f"get_feature_names_out(None) after fit() != originally passed columns"
        del _ACTIVE_COLUMNS

        # WITH HEADER PASSED, SHOULD RAISE TypeError IF input_features
        # FOR DISALLOWED TYPES

        JUNK_COL_NAMES = [
            [*range(len(_columns))], [*range(2 * len(_columns))], {'a': 1, 'b': 2}
        ]
        for junk_col_names in JUNK_COL_NAMES:
            with pytest.raises(TypeError):
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

        assert np.array_equiv(RETURNED_FROM_GFNO,
                      np.array(_columns)[TestCls.get_support(False)]), \
            f"get_feature_names_out() did not return original columns"

        del junk_col_names, RETURNED_FROM_GFNO
        # END ^^^ COLUMN NAMES PASSED (PD) ^^^

        # END get_feature_names_out() **********************************

        # get_metadata_routing()
        with pytest.raises(NotImplementedError):
            TestCls.get_metadata_routing()

        # get_params()
        TestCls.get_params(True)

        # get_row_support()
        with pytest.raises(AttributeError):
            TestCls.get_row_support(False)

        # get_support()
        for _indices in [True, False]:
            __ = TestCls.get_support(_indices)
            assert isinstance(__, np.ndarray), \
                f"get_support() did not return numpy.ndarray"

            if not _indices:
                assert __.dtype == 'bool', \
                    f"get_support with indices=False did not return a boolean array"
                assert len(__) == TestCls.n_features_in_, \
                    f"len(get_support(False)) != n_features_in_"
                assert sum(__) == len(TestCls.get_feature_names_out(None))
            elif _indices:
                assert 'int' in str(__.dtype).lower(), \
                    (f"get_support with indices=True did not return an array of "
                     f"integers")
                assert len(__) == len(TestCls.get_feature_names_out(None))

        del TestCls, _indices, __,

        # ** _handle_X_y()

        # inverse_transform() ********************
        TestCls = CDT(**_kwargs)
        TestCls.fit(_dum_X, y)  # X IS NP ARRAY

        # SHOULD RAISE ValueError IF X IS NOT A 2D ARRAY
        for junk_x in [[], [[]]]:
            with pytest.raises(ValueError):
                TestCls.inverse_transform(junk_x)

        # SHOULD RAISE TypeError IF X IS NOT A LIST-TYPE
        for junk_x in [None, 'junk_string', 3, np.pi]:
            with pytest.raises(TypeError):
                TestCls.inverse_transform(junk_x)

        # SHOULD RAISE ValueError WHEN COLUMNS DO NOT EQUAL NUMBER OF
        # RETAINED COLUMNS
        TRFM_X = TestCls.transform(X)
        TRFM_MASK = TestCls.get_support(False)
        __ = np.array(_columns)
        for obj_type in ['np', 'pd']:
            for diff_cols in ['more', 'less', 'same']:
                if diff_cols == 'same':
                    TEST_X = TRFM_X.copy()
                    if obj_type == 'pd':
                        TEST_X = pd.DataFrame(data=TEST_X, columns=__[TRFM_MASK])
                elif diff_cols == 'less':
                    TEST_X = TRFM_X[:, :2].copy()
                    if obj_type == 'pd':
                        TEST_X = pd.DataFrame(data=TEST_X, columns=__[TRFM_MASK][:2])
                elif diff_cols == 'more':
                    TEST_X = np.hstack((TRFM_X.copy(), TRFM_X.copy()))
                    if obj_type == 'pd':
                        _COLUMNS = np.hstack((
                            __[TRFM_MASK],
                            np.char.upper(__[TRFM_MASK])
                        ))
                        TEST_X = pd.DataFrame(data=TEST_X, columns=_columns)

                if diff_cols == 'same':
                    TestCls.inverse_transform(TEST_X)
                else:
                    with pytest.raises(ValueError):
                        TestCls.inverse_transform(TEST_X)

        INV_TRFM_X = TestCls.inverse_transform(TRFM_X)

        assert isinstance(INV_TRFM_X, np.ndarray), \
            f"output of inverse_transform() is not a numpy array"
        assert INV_TRFM_X.shape[0] == TRFM_X.shape[0], \
            f"rows in output of inverse_transform() do not match input rows"
        assert INV_TRFM_X.shape[1] == TestCls.n_features_in_, \
            (f"columns in output of inverse_transform() do not match "
             f"originally fitted columns")

        __ = np.logical_not(TestCls.get_support(False))
        assert np.array_equiv(INV_TRFM_X[:, __],
                              np.zeros((TRFM_X.shape[0], sum(__)))
            ), \
            (f"back-filled parts of inverse_transform() output do not slice "
             f"to a zero array")
        del __

        assert np.array_equiv(
            TRFM_X.astype(str),
            INV_TRFM_X[:, TestCls.get_support(False)].astype(str)
            ), (f"output of inverse_transform() does not reduce back to "
                f"the output of transform()")

        del junk_x, TRFM_X, TRFM_MASK, obj_type, diff_cols
        del TEST_X, INV_TRFM_X, TestCls

        # END inverse_transform() **********

        TestCls = CDT(**_kwargs)

        # ** _make_instructions()
        # ** _must_be_fitted()
        # partial_fit()
        # ** _reset()

        # set_output()
        TestCls.set_output(transform='pandas_dataframe')

        # set_params()
        KEYS = [
            'count_threshold', 'ignore_float_columns',
            'ignore_non_binary_integer_columns', 'ignore_columns', 'ignore_nan',
            'handle_as_bool', 'delete_axis_0', 'reject_unseen_values',
            'max_recursions', 'n_jobs'
        ]
        VALUES = [4, False, False, [0], False, [2], True, True, 2, 4]
        test_kwargs = dict((zip(KEYS, VALUES)))

        TestCls.set_params(**test_kwargs)
        ATTRS = [
            TestCls._count_threshold, TestCls._ignore_float_columns,
            TestCls._ignore_non_binary_integer_columns, TestCls._ignore_columns,
            TestCls._ignore_nan, TestCls._handle_as_bool, TestCls._delete_axis_0,
            TestCls._reject_unseen_values, TestCls._max_recursions,
            TestCls._n_jobs
        ]
        for _key, _attr, _value in zip(KEYS, ATTRS, VALUES):
            assert _attr == _value, f'set_params() did not set {_key}'

        del TestCls, KEYS, VALUES, ATTRS

        # transform()
        # ** _validate_delete_instr()
        # ** _validate_feature_names()
        # ** _validate()

        # END ^^^ AFTER FIT ^^^ ****************************************
        # **************************************************************

    def test_access_methods_after_transform(self, _X, _columns, _kwargs, _shape):

        X = _X()    # pizza figure out what this needs to be
        y = np.random.randint(0, 2, _shape[0])

        # **************************************************************
        # vvv AFTER TRANSFORM vvv **************************************
        FittedTestCls = CDT(**_kwargs).fit(X, y)
        TransformedTestCls = CDT(**_kwargs).fit(X, y)
        TRFM_X = TransformedTestCls.transform(X, y)

        # ** _base_fit()
        # ** _check_is_fitted()

        # ** test_threshold()
        # SHOULD BE THE SAME AS AFTER FIT
        TransformedTestCls.test_threshold()
        print(f'^^^ mask building instructions should be displayed above ^^^')

        # fit()
        # fit_transform()

        # get_feature_names_out() **************************************
        # vvv NO COLUMN NAMES PASSED (NP) vvv

        # # WHEN NO HEADER PASSED TO (partial_)fit() AND VALID input_features,
        # SHOULD RETURN ORIGINAL (UNSLICED) COLUMNS
        RETURNED_FROM_GFNO = TransformedTestCls.get_feature_names_out(_columns)


        _ACTIVE_COLUMNS = np.array(_columns)[TransformedTestCls.get_support(False)]
        assert np.array_equiv(RETURNED_FROM_GFNO, _ACTIVE_COLUMNS), \
            (f"get_feature_names_out() after transform did not return "
             f"sliced original columns")

        del RETURNED_FROM_GFNO
        # END ^^^ NO COLUMN NAMES PASSED (NP) ^^^

        # vvv COLUMN NAMES PASSED (PD) vvv
        PDTransformedTestCls = CDT(**_kwargs)
        PDTransformedTestCls.fit_transform(pd.DataFrame(data=X, columns=_columns), y)

        # WITH HEADER PASSED AND input_features=None,
        # SHOULD RETURN SLICED ORIGINAL COLUMNS
        assert np.array_equiv(PDTransformedTestCls.get_feature_names_out(None),
                      np.array(_columns)[PDTransformedTestCls.get_support(False)]), \
            (f"get_feature_names_out(None) after transform() != "
             f"originally passed columns")

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

        # get_row_support()
        for _indices in [True, False]:
            __ = TransformedTestCls.get_row_support(_indices)
            assert isinstance(__, np.ndarray), \
                f"get_row_support() did not return numpy.ndarray"

            if not _indices:
                assert __.dtype == 'bool', \
                    (f"get_row_support with indices=False did not return a "
                     f"boolean array")
            elif _indices:
                assert 'int' in str(__.dtype).lower(), \
                    (f"get_row_support with indices=True did not return an "
                     f"array of integers")

        del __

        # get_support()
        assert np.array_equiv(FittedTestCls.get_support(False),
            TransformedTestCls.get_support(False)), \
            f"get_support(False) after transform() != get_support(False) before"

        # ** _handle_X_y()

        # inverse_transform() ************

        assert np.array_equiv(
            FittedTestCls.inverse_transform(TRFM_X).astype(str),
            TransformedTestCls.inverse_transform(TRFM_X).astype(str)), \
            (f"inverse_transform(TRFM_X) after transform() != "
             f"inverse_transform(TRFM_X) before transform()")

        # END inverse_transform() **********

        # ** _make_instructions()
        # ** _must_be_fitted()
        # partial_fit()
        # ** _reset()

        # set_output()
        TransformedTestCls.set_output(transform='pandas_dataframe')
        TransformedTestCls.transform(X, y)

        del TransformedTestCls

        # set_params()
        TestCls = CDT(**_kwargs)
        KEYS = [
            'count_threshold', 'ignore_float_columns',
            'ignore_non_binary_integer_columns', 'ignore_columns', 'ignore_nan',
            'handle_as_bool', 'delete_axis_0', 'reject_unseen_values',
            'max_recursions', 'n_jobs'
        ]
        VALUES = [4, False, False, [0], False, [2], True, True, 2, 4]
        test_kwargs = dict((zip(KEYS, VALUES)))

        TestCls.set_params(**test_kwargs)
        TestCls.fit_transform(X, y)
        ATTRS = [
            TestCls._count_threshold, TestCls._ignore_float_columns,
            TestCls._ignore_non_binary_integer_columns, TestCls._ignore_columns,
            TestCls._ignore_nan, TestCls._handle_as_bool, TestCls._delete_axis_0,
            TestCls._reject_unseen_values, TestCls._max_recursions,
            TestCls._n_jobs
        ]
        for _key, _attr, _value in zip(KEYS, ATTRS, VALUES):
            assert _attr == _value, f'set_params() did not set {_key}'

        del KEYS, VALUES, ATTRS

        # transform()
        # ** _validate_delete_instr()
        # ** _validate_feature_names()
        # ** _validate()

        del FittedTestCls, TestCls, TRFM_X

        # END ^^^ AFTER TRANSFORM ^^^ **********************************
        # **************************************************************

# END ACCESS METHODS BEFORE AND AFTER FIT AND TRANSFORM


# ACCESS METHODS BEFORE AND AFTER FIT AND TRANSFORM; FOR 2 RECURSIONS **
@pytest.mark.skipif(bypass is True, reason=f"bypass")
class Test2RecursionAccessMethodsBeforeAndAfterFitAndTransform:

    # CREATE AN INSTANCE WITH ONLY 1 RECURSION TO COMPARE 1X-TRFMED OBJECTS
    # AGAINST 2X-TRFMED OBJECTS
    @staticmethod
    @pytest.fixture
    def OneRecurTestCls(_kwargs):
        args = [3]
        _kwargs['ignore_columns'] = None
        _kwargs['ignore_nan'] = False
        _kwargs['ignore_non_binary_integer_columns'] = False
        _kwargs['ignore_float_columns'] = False
        _kwargs['delete_axis_0'] = True
        _kwargs['max_recursions'] = 1

        return CDT(**_kwargs)


    @staticmethod
    @pytest.fixture
    def TwoRecurTestCls(_kwargs):
        args = [3]
        _kwargs['ignore_columns'] = None
        _kwargs['ignore_nan'] = False
        _kwargs['ignore_non_binary_integer_columns'] = False
        _kwargs['ignore_float_columns'] = False
        _kwargs['delete_axis_0'] = True
        _kwargs['max_recursions'] = 2

        return CDT(**_kwargs)

    def test_before_fit_transform(self, OneRecurTestCls, TwoRecurTestCls,
        X, y, _args, _kwargs):

        # **************************************************************
        # vvv BEFORE fit_transform() vvv *******************************

        # ** _base_fit()
        # ** _check_is_fitted()

        # ** test_threshold()
        with pytest.raises(AttributeError):
            TwoRecurTestCls.test_threshold()

        # fit()
        # fit_transform()

        # get_feature_names_out()
        with pytest.raises(AttributeError):
            TwoRecurTestCls.get_feature_names_out(None)

        # get_metadata_routing()
        with pytest.raises(NotImplementedError):
            TwoRecurTestCls.get_metadata_routing()

        # get_params()
        # ALL PARAMS SHOULD BE THE SAME EXCEPT FOR max_recursions
        _ = OneRecurTestCls.get_params(True)
        del _['max_recursions']
        __ = TwoRecurTestCls.get_params(True)
        del __['max_recursions']
        assert _ == __, (f"pre-fit 1 recursion instance get_params() != "
                         f"get_params() from 2 recursion instance")
        del _, __

        # get_row_support()
        with pytest.raises(AttributeError):
            TwoRecurTestCls.get_row_support(True)

        # get_support()
        with pytest.raises(AttributeError):
            TwoRecurTestCls.get_support(True)

        # ** _handle_X_y()

        # inverse_transform()
        with pytest.raises(AttributeError):
            TwoRecurTestCls.inverse_transform(X)

        # ** _make_instructions()
        # ** _must_be_fitted()
        # partial_fit()
        # ** _reset()

        # set_output()
        TwoRecurTestCls.set_output(transform='pandas_dataframe')

        # set_params()
        TestCls = CDT(**_kwargs)
        KEYS = [
            'count_threshold', 'ignore_float_columns',
            'ignore_non_binary_integer_columns', 'ignore_columns', 'ignore_nan',
            'handle_as_bool', 'delete_axis_0', 'reject_unseen_values',
            'max_recursions', 'n_jobs'
        ]
        VALUES = [4, False, False, [0], False, [2], True, True, 2, 4]
        test_kwargs = dict((zip(KEYS, VALUES)))

        TestCls.set_params(**test_kwargs)
        ATTRS = [
            TestCls._count_threshold, TestCls._ignore_float_columns,
            TestCls._ignore_non_binary_integer_columns, TestCls._ignore_columns,
            TestCls._ignore_nan, TestCls._handle_as_bool, TestCls._delete_axis_0,
            TestCls._reject_unseen_values, TestCls._max_recursions, TestCls._n_jobs
        ]

        for _key, _attr, _value in zip(KEYS, ATTRS, VALUES):
            assert _attr == _value, f'set_params() did not set {_key}'

        del TestCls, KEYS, VALUES, ATTRS

        TwoRecurTestCls = CDT(**_kwargs)
        # transform()
        with pytest.raises(AttributeError):
            TwoRecurTestCls.transform(X, y)

        # ** _validate_delete_instr()
        # ** _validate_feature_names()
        # ** _validate()

        # END ^^^ BEFORE fit_transform() ^^^ ***************************
        # **************************************************************

        del TwoRecurTestCls


    def test_after_fit_transform(self, OneRecurTestCls, TwoRecurTestCls, _X,
        _columns, _kwargs, _shape):

        X = _X()   # pizza figure out what this needs to be
        y = np.random.randint(0,2,_shape[0])

        ONE_RCR_TRFM_X, ONE_RCR_TRFM_Y = OneRecurTestCls.fit_transform(X, y)
        TWO_RCR_TRFM_X, TWO_RCR_TRFM_Y = TwoRecurTestCls.fit_transform(X, y)

        # **************************************************************
        # vvv AFTER fit_transform() vvv ********************************

        # ** _base_fit()
        # ** _check_is_fitted()

        # ** test_threshold()
        assert not np.array_equiv(ONE_RCR_TRFM_X, TWO_RCR_TRFM_X), \
            f"ONE_RCR_TRFM_X == TWO_RCR_TRFM_X when it shouldnt"

        TwoRecurTestCls.test_threshold(clean_printout=True)
        print(f'^^^ mask building instructions should be displayed above ^^^')


        # fit()
        # fit_transform()

        # get_feature_names_out() **************************************
        # vvv NO COLUMN NAMES PASSED (NP) vvv

        # WITH NO HEADER PASSED AND input_features=None, SHOULD RETURN
        # SLICED ['x0', ..., 'x(n-1)]
        _COLUMNS = np.array([f"x{i}" for i in range(len(_columns))])
        _ACTIVE_COLUMNS = _COLUMNS[TwoRecurTestCls.get_support(False)]
        del _COLUMNS
        assert np.array_equiv(
            TwoRecurTestCls.get_feature_names_out(None),
            _ACTIVE_COLUMNS
        ), (f"get_feature_names_out(None) after fit_transform() != sliced "
            f"array of generic headers"
        )
        del _ACTIVE_COLUMNS

        # WITH NO HEADER PASSED, SHOULD RAISE ValueError IF len(input_features) !=
        # n_features_in_
        with pytest.raises(ValueError):
            _COLUMNS = [f"x{i}" for i in range(2 * len(_columns))]
            TwoRecurTestCls.get_feature_names_out(_COLUMNS)
            del _COLUMNS

        # WHEN NO HEADER PASSED TO fit_transform() AND VALID input_features,
        # SHOULD RETURN SLICED PASSED COLUMNS
        RETURNED_FROM_GFNO = TwoRecurTestCls.get_feature_names_out(_columns)
        assert isinstance(RETURNED_FROM_GFNO, np.ndarray), \
            (f"TwoRecur.get_feature_names_out should return numpy.ndarray, "
             f"but returned {type(RETURNED_FROM_GFNO)}")

        assert np.array_equiv(RETURNED_FROM_GFNO,
            np.array(_columns)[TwoRecurTestCls.get_support(False)]), \
            f"TwoRecur.get_feature_names_out() did not return original columns"

        del RETURNED_FROM_GFNO

        # END ^^^ NO COLUMN NAMES PASSED (NP) ^^^

        # vvv COLUMN NAMES PASSED (PD) vvv
        ONE_RCR_TRFM_X, ONE_RCR_TRFM_Y = \
            OneRecurTestCls.fit_transform(pd.DataFrame(data=X, columns=_columns), y)

        TWO_RCR_TRFM_X, TWO_RCR_TRFM_Y = \
            TwoRecurTestCls.fit_transform(pd.DataFrame(data=X, columns=_columns), y)

        # WITH HEADER PASSED AND input_features=None:
        # SHOULD RETURN SLICED ORIGINAL COLUMNS
        assert np.array_equiv(
            TwoRecurTestCls.get_feature_names_out(None),
            np.array(_columns)[TwoRecurTestCls.get_support(False)]
            ), (f"TwoRecur.get_feature_names_out(None) after fit_transform() != "
            f"sliced originally passed columns"
        )

        # WHEN HEADER PASSED TO fit_transform() AND input_features IS THAT HEADER,
        # SHOULD RETURN SLICED VERSION OF THAT HEADER
        RETURNED_FROM_GFNO = TwoRecurTestCls.get_feature_names_out(_columns)
        assert isinstance(RETURNED_FROM_GFNO, np.ndarray), \
            (f"get_feature_names_out should return numpy.ndarray, but returned "
             f"{type(RETURNED_FROM_GFNO)}")
        assert np.array_equiv(
            RETURNED_FROM_GFNO,
            np.array(_columns)[TwoRecurTestCls.get_support(False)]
            ), f"get_feature_names_out() did not return original columns"

        del RETURNED_FROM_GFNO
        # END ^^^ COLUMN NAMES PASSED (PD) ^^^
        # END get_feature_names_out() **********************************

        # get_metadata_routing()
        with pytest.raises(NotImplementedError):
            TwoRecurTestCls.get_metadata_routing()

        # get_params()
        # ALL PARAMS SHOULD BE THE SAME EXCEPT FOR max_recursions
        _ = OneRecurTestCls.get_params(True)
        del _['max_recursions']
        __ = TwoRecurTestCls.get_params(True)
        del __['max_recursions']
        assert _ == __, (f"pre-fit 1 recursion instance get_params() != "
                         f"get_params() from 2 recursion instance")
        del _, __

        # get_row_support()
        for _indices in [True, False]:
            _ONE = OneRecurTestCls.get_row_support(_indices)
            _TWO = TwoRecurTestCls.get_row_support(_indices)

            assert isinstance(_ONE, np.ndarray), \
                f"get_row_support() for 1 recursion did not return numpy.ndarray"
            assert isinstance(_TWO, np.ndarray), \
                f"get_row_support() for 2 recursions did not return numpy.ndarray"

            if not _indices:
                assert _ONE.dtype == 'bool', (f"get_row_support with indices=False "
                          f"for 1 recursion did not return a boolean array")
                assert _TWO.dtype == 'bool', (f"get_row_support with indices=False "
                              f"for 2 recursions did not return a boolean array")
                # len(ROW SUPPORT TWO RECUR) AND len(ROW SUPPORT ONE RECUR)
                # MUST EQUAL NUMBER OF ROWS IN X
                assert len(_ONE) == _shape[0], \
                    (f"row_support vector length for 1 recursion != rows in "
                     f"passed data"
                )
                assert len(_TWO) == _shape[0], \
                    (f"row_support vector length for 2 recursions != rows in "
                     f"passed data"
                )
                # NUMBER OF Trues in ONE RECUR MUST == NUMBER OF ROWS IN
                # ONE RCR TRFM X; SAME FOR TWO RCR
                assert sum(_ONE) == ONE_RCR_TRFM_X.shape[0], \
                    f"one rcr Trues IN row_support != TRFM X rows"
                assert sum(_TWO) == TWO_RCR_TRFM_X.shape[0], \
                    f"two rcr Trues IN row_support != TRFM X rows"
                # NUMBER OF Trues IN ONE RECUR MUST BE >= NUMBER OF Trues
                # IN TWO RECUR
                assert sum(_ONE) >= sum(_TWO), \
                    f"two recursion has more rows kept in it that one recursion"
                # ANY Trues IN TWO RECUR MUST ALSO BE True IN ONE RECUR
                assert np.unique(_ONE[_TWO])[0] == True, \
                    (f"Rows that are to be kept in 2nd recur (True) were False "
                     f"in 1st recur")
            elif _indices:
                assert 'int' in str(_ONE.dtype).lower(), \
                    (f"get_row_support with indices=True for 1 recursion did not "
                     f"return an array of integers")
                assert 'int' in str(_TWO.dtype).lower(), \
                    (f"get_row_support with indices=True for 2 recursions did not "
                     f"return an array of integers")
                # len(row_support) ONE RECUR MUST == NUMBER OF ROWS IN ONE RCR
                # TRFM X; SAME FOR TWO RCR
                assert len(_ONE) == ONE_RCR_TRFM_X.shape[0], \
                    f"one rcr len(row_support) as idxs does not equal TRFM X rows"
                assert len(_TWO) == TWO_RCR_TRFM_X.shape[0], \
                    f"two rcr len(row_support) as idxs does not equal TRFM X rows "
                # NUMBER OF ROW IDXS IN ONE RECUR MUST BE >= NUM ROW IDXS IN TWO RECUR
                assert len(_ONE) >= len(_TWO), \
                    f"two recursion has more rows kept in it that one recursion"
                # INDICES IN TWO RECUR MUST ALSO BE IN ONE RECUR
                for row_idx in _TWO:
                    assert row_idx in _ONE, (f"Rows that are to be kept by 2nd "
                                             f"recur were not kept by 1st recur")

        del _ONE, _TWO, row_idx, _indices
        del ONE_RCR_TRFM_X, ONE_RCR_TRFM_Y, TWO_RCR_TRFM_X, TWO_RCR_TRFM_Y

        # get_support()
        for _indices in [True, False]:
            _ = OneRecurTestCls.get_support(_indices)
            __ = TwoRecurTestCls.get_support(_indices)
            assert isinstance(_, np.ndarray), \
                f"2 recursion get_support() did not return numpy.ndarray"
            assert isinstance(__, np.ndarray), \
                f"2 recursion get_support() did not return numpy.ndarray"

            if not _indices:
                assert _.dtype == 'bool', \
                    (f"1 recursion get_support with indices=False did not "
                     f"return a boolean array")
                assert __.dtype == 'bool', \
                    (f"2 recursion get_support with indices=False did not "
                     f"return a boolean array")

                # len(ROW SUPPORT TWO RECUR) AND len(ROW SUPPORT ONE RECUR)
                # MUST EQUAL NUMBER OF COLUMNS IN X
                assert len(_) == _shape[1], \
                    f"1 recursion len(get_support({_indices})) != X columns"
                assert len(__) == _shape[1], \
                    f"2 recursion len(get_support({_indices})) != X columns"
                # NUM COLUMNS IN 1 RECURSION MUST BE <= NUM COLUMNS IN X
                assert sum(_) <= _shape[1], \
                    (f"impossibly, number of columns kept by 1 recursion > number "
                     f"of columns in X")
                # NUM COLUMNS IN 2 RECURSION MUST BE <= NUM COLUMNS IN 1 RECURSION
                assert sum(__) <= sum(_),\
                    (f"impossibly, number of columns kept by 2 recursion > number "
                     f"of columns kept by 1 recursion")
                # INDICES IN TWO RECUR MUST ALSO BE IN ONE RECUR
                assert np.unique(_[__])[0] == True, (f"Columns that are to be "
                         f"kept in 2nd recur (True) were False in 1st recur")
            elif _indices:
                assert 'int' in str(_.dtype).lower(), (f"1 recursion get_support "
                    f"with indices=True did not return an array of integers")
                assert 'int' in str(__.dtype).lower(), (f"2 recursion get_support "
                    f"with indices=True did not return an array of integers")
                # ONE RECURSION COLUMNS MUST BE <= n_features_in_
                assert len(_) <= _shape[1], \
                    f"impossibly, 1 recursion len(get_support({_indices})) > X columns"
                # TWO RECURSION COLUMNS MUST BE <= ONE RECURSION COLUMNS
                assert len(__) <= len(_), \
                    (f"2 recursion len(get_support({_indices})) > 1 "
                     f"recursion len(get_support({_indices}))")
                # INDICES IN TWO RECUR MUST ALSO BE IN ONE RECUR
                for col_idx in __:
                    assert col_idx in _, (f"Columns that are to be kept by "
                              f"2nd recur were not kept by 1st recur")

        del TwoRecurTestCls, _, __, _indices, col_idx

        # ** _handle_X_y()

        # inverse_transform() ********************
        TwoRecurTestCls = CDT(**_kwargs)
        # X IS NP ARRAY
        TRFM_X, TRFM_Y = TwoRecurTestCls.fit_transform(X, y)

        # SHOULD RAISE ValueError WHEN COLUMNS DO NOT EQUAL NUMBER OF RETAINED COLUMNS
        __ = np.array(_columns)
        TRFM_MASK = TwoRecurTestCls.get_support(False)
        for obj_type in ['np', 'pd']:
            for diff_cols in ['more', 'less', 'same']:
                if diff_cols == 'same':
                    TEST_X = TRFM_X.copy()
                    if obj_type == 'pd':
                        TEST_X = pd.DataFrame(data=TEST_X, columns=__[TRFM_MASK])
                elif diff_cols == 'less':
                    TEST_X = TRFM_X[:, :2].copy()
                    if obj_type == 'pd':
                        TEST_X = pd.DataFrame(data=TEST_X, columns=__[TRFM_MASK][:2])
                elif diff_cols == 'more':
                    TEST_X = np.hstack((TRFM_X.copy(), TRFM_X.copy()))
                    if obj_type == 'pd':
                        _COLUMNS = np.hstack((__[TRFM_MASK],
                                              np.char.upper(__[TRFM_MASK])
                        ))
                        TEST_X = pd.DataFrame(data=TEST_X, columns=_COLUMNS)
                        del _COLUMNS

                if diff_cols == 'same':
                    TwoRecurTestCls.inverse_transform(TEST_X)
                else:
                    with pytest.raises(ValueError):
                        TwoRecurTestCls.inverse_transform(TEST_X)

        INV_TRFM_X = TwoRecurTestCls.inverse_transform(TRFM_X)

        assert isinstance(INV_TRFM_X, np.ndarray), \
            f"output of inverse_transform() is not a numpy array"
        assert INV_TRFM_X.shape[0] == TRFM_X.shape[0], \
            f"rows in output of inverse_transform() do not match input rows"
        assert INV_TRFM_X.shape[1] == TwoRecurTestCls.n_features_in_, \
            (f"columns in output of inverse_transform() do not match originally "
             f"fitted columns")

        __ = np.logical_not(TwoRecurTestCls.get_support(False))
        _ZERO_ARRAY = np.zeros((TRFM_X.shape[0], sum(__)))
        assert np.array_equiv(INV_TRFM_X[:, __], _ZERO_ARRAY), (f"back-filled "
            f"parts of inverse_transform() output do not slice to a zero array")
        del __, _ZERO_ARRAY

        assert np.array_equiv(TRFM_X.astype(str),
            INV_TRFM_X[:, TwoRecurTestCls.get_support(False)].astype(str)), \
            (f"output of inverse_transform() does not reduce back to the output "
             f"of transform()")

        del TRFM_X, TRFM_Y, TRFM_MASK, obj_type, diff_cols, TEST_X, INV_TRFM_X

        # END inverse_transform() **********

        # ** _make_instructions()
        # ** _must_be_fitted()
        # partial_fit()
        # ** _reset()

        # set_output()
        TwoRecurTestCls.set_output(transform='pandas_dataframe')
        assert TwoRecurTestCls._output_transform == 'pandas_dataframe'
        TwoRecurTestCls.fit_transform(X, y)
        assert TwoRecurTestCls._output_transform == 'pandas_dataframe'

        del TwoRecurTestCls

        # set_params()
        TestCls = CDT(**_kwargs)
        KEYS = [
            'count_threshold', 'ignore_float_columns',
            'ignore_non_binary_integer_columns', 'ignore_columns', 'ignore_nan',
            'handle_as_bool', 'delete_axis_0', 'reject_unseen_values',
            'max_recursions', 'n_jobs'
        ]
        VALUES = [4, False, False, [0], False, [2], True, True, 2, 4]
        test_kwargs = dict((zip(KEYS, VALUES)))

        TestCls.set_params(**test_kwargs)
        ATTRS = [
            TestCls._count_threshold, TestCls._ignore_float_columns,
            TestCls._ignore_non_binary_integer_columns, TestCls._ignore_columns,
            TestCls._ignore_nan, TestCls._handle_as_bool, TestCls._delete_axis_0,
            TestCls._reject_unseen_values, TestCls._max_recursions, TestCls._n_jobs
        ]
        for _key, _attr, _value in zip(KEYS, ATTRS, VALUES):
            assert _attr == _value, f'set_params() did not set {_key}'

        del TestCls, KEYS, VALUES, ATTRS

        # transform()
        # ** _validate_delete_instr()
        # ** _validate_feature_names()
        # ** _validate()

        del OneRecurTestCls,

        # END ^^^ AFTER fit_transform() ^^^ ****************************
        # **************************************************************


# END ACCESS METHODS BEFORE AND AFTER FIT AND TRANSFORM; FOR 2 RECURSIONS
