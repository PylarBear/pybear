# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pytest

import numpy as np
import pandas as pd

from uuid import uuid4

from pybear.base.exceptions import NotFittedError

from pybear.preprocessing.MinCountTransformer.MinCountTransformer import \
    MinCountTransformer as MCT




bypass = False



@pytest.fixture(scope='function')
def _rows():
    return 200


@pytest.fixture(scope='function')
def _cols():
    return 5


@pytest.fixture(scope='function')
def _args():
    return [5]


@pytest.fixture(scope='function')
def _kwargs():
    return {
        'ignore_float_columns': False,
        'ignore_non_binary_integer_columns': False,
        'ignore_columns': None,
        'ignore_nan': False,
        'delete_axis_0': False,
        'handle_as_bool': None,
        'reject_unseen_values': True,
        'max_recursions': 1,
        'n_jobs': -1
    }



@pytest.fixture(scope='function')
def X(_args, _rows, _cols):
    return np.random.randint(0, _rows//_args[0], (_rows, _cols))



@pytest.fixture(scope='function')
def y(_rows):
    return np.random.randint(0, 2, (_rows, 2))



@pytest.fixture(scope='function')
def COLUMNS(_cols):
    return [str(uuid4())[:5] for _ in range(_cols)]








# ACCESS ATTR BEFORE AND AFTER FIT AND TRANSFORM, ATTR ACCURACY; FOR 1 RECURSION
@pytest.mark.skipif(bypass is True, reason=f"bypass")
class TestAttrAccuracyBeforeAndAfterFitAndTransform:

    def test_attr_accuracy(self, X, COLUMNS, y, _args, _kwargs):

        NEW_X = X.copy()
        NEW_Y = y.copy()
        NEW_X_DF = pd.DataFrame(data=X, columns=COLUMNS)
        NEW_Y_DF = pd.DataFrame(data=y, columns=['y1', 'y2'])

        # BEFORE FIT ***************************************************

        TestCls = MCT(*_args, **_kwargs)

        # ALL OF THESE SHOULD GIVE AttributeError
        with pytest.raises(AttributeError):
            TestCls.feature_names_in_

        with pytest.raises(AttributeError):
            TestCls.n_features_in_

        with pytest.raises(AttributeError):
            TestCls.original_dtypes_

        with pytest.raises(AttributeError):
            TestCls.original_dtypes_ = list('abcde')

        del TestCls
        # END BEFORE FIT ***********************************************

        # AFTER FIT ****************************************************
        for data_dtype in ['np', 'pd']:
            if data_dtype == 'np':
                TEST_X, TEST_Y = NEW_X.copy(), NEW_Y.copy()
            elif data_dtype == 'pd':
                TEST_X, TEST_Y = NEW_X_DF.copy(), NEW_Y_DF.copy()

            TestCls = MCT(*_args, **_kwargs)
            TestCls.fit(TEST_X, TEST_Y)

            # ONLY EXCEPTION SHOULD BE feature_names_in_ IF NUMPY
            if data_dtype == 'pd':
                assert np.array_equiv(TestCls.feature_names_in_, COLUMNS), \
                    f"feature_names_in_ after fit() != originally passed columns"
            elif data_dtype == 'np':
                with pytest.raises(AttributeError):
                    TestCls.feature_names_in_

            assert TestCls.n_features_in_ == X.shape[1], \
                f"n_features_in_ after fit() != number of originally passed columns"

            assert np.array_equiv(
                TestCls._original_dtypes,
                ['int' for _ in range(X.shape[1])]
            ), f"_original_dtypes after fit() != originally passed dtypes"

        del data_dtype, TEST_X, TEST_Y, TestCls

        # END AFTER FIT ************************************************

        # AFTER TRANSFORM **********************************************

        for data_dtype in ['np', 'pd']:

            if data_dtype == 'np':
                TEST_X, TEST_Y = NEW_X.copy(), NEW_Y.copy()
            elif data_dtype == 'pd':
                TEST_X, TEST_Y = NEW_X_DF.copy(), NEW_Y_DF.copy()

            TestCls = MCT(*_args, **_kwargs)
            TestCls.fit_transform(TEST_X, TEST_Y)

            # ONLY EXCEPTION SHOULD BE feature_names_in_ WHEN NUMPY
            if data_dtype == 'pd':
                assert np.array_equiv(TestCls.feature_names_in_, COLUMNS), \
                    f"feature_names_in_ after fit() != originally passed columns"
            elif data_dtype == 'np':
                with pytest.raises(AttributeError):
                    TestCls.feature_names_in_

            assert TestCls.n_features_in_ == X.shape[1], \
                f"n_features_in_ after fit() != number of originally passed columns"

            assert np.array_equiv(
                TestCls._original_dtypes,
                ['int' for _ in range(X.shape[1])]
            ), f"_original_dtypes after fit() != originally passed dtypes"

        del data_dtype, TEST_X, TEST_Y, TestCls
        # END AFTER TRANSFORM ******************************************

        del NEW_X, NEW_Y, NEW_X_DF, NEW_Y_DF

# END ACCESS ATTR BEFORE AND AFTER FIT AND TRANSFORM, ATTR ACCURACY; FOR 1 RECURSION


# ACCESS METHODS BEFORE AND AFTER FIT AND TRANSFORM; FOR 1 RECURSION ***
@pytest.mark.skipif(bypass is True, reason=f"bypass")
class Test1RecursionAccessMethodsBeforeAndAfterFitAndTransform:

    def test_access_methods_before_fit(self, X, y, _args, _kwargs):

        TestCls = MCT(*_args, **_kwargs)

        # **************************************************************
        # vvv BEFORE FIT vvv *******************************************

        # ** _base_fit()
        # ** _check_is_fitted()

        # ** test_threshold()
        with pytest.raises(NotFittedError):
            TestCls.test_threshold()

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

        # ** _handle_X_y()

        # inverse_transform()
        with pytest.raises(NotFittedError):
            TestCls.inverse_transform(X)

        # ** _make_instructions()
        # ** _must_be_fitted()
        # partial_fit()
        # ** _reset()

        # set_output()
        TestCls.set_output(transform='pandas_dataframe')

        # set_params()
        TestCls.set_params(count_threshold=5)

        del TestCls

        TestCls = MCT(*_args, **_kwargs)
        # transform()
        with pytest.raises(NotFittedError):
            TestCls.transform(X, y)

        # ** _validate_delete_instr()
        # ** _validate_feature_names()
        # ** _validate()

        # END ^^^ BEFORE FIT ^^^ ***************************************
        # **************************************************************

    def test_access_methods_after_fit(self, X, COLUMNS, y, _args, _kwargs):

        # **************************************************************
        # vvv AFTER FIT vvv ********************************************

        TestCls = MCT(*_args, **_kwargs)
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
                     [*range(len(COLUMNS))]
        ]

        for junk_arg in JUNK_ARGS:
            with pytest.raises(ValueError):
                TestCls.get_feature_names_out(junk_arg)

        del JUNK_ARGS

        # WITH NO HEADER PASSED AND input_features=None, SHOULD RETURN
        # ['x0', ..., 'x(n-1)][COLUMN MASK]
        _COLUMNS = np.array([f"x{i}" for i in range(len(COLUMNS))])
        ACTIVE_COLUMNS = _COLUMNS[TestCls.get_support(False)]
        del _COLUMNS
        assert np.array_equiv(
            TestCls.get_feature_names_out(None),
            ACTIVE_COLUMNS
        ), (f"get_feature_names_out(None) after fit() != sliced array of "
            f"generic headers")
        del ACTIVE_COLUMNS

        # WITH NO HEADER PASSED, SHOULD RAISE ValueError IF
        # len(input_features) != n_features_in_
        with pytest.raises(ValueError):
            TestCls.get_feature_names_out(
                [f"x{i}" for i in range(2 * len(COLUMNS))]
            )

        # WHEN NO HEADER PASSED TO (partial_)fit() AND VALID input_features,
        # SHOULD RETURN SLICED PASSED COLUMNS
        RETURNED_FROM_GFNO = TestCls.get_feature_names_out(COLUMNS)
        assert isinstance(RETURNED_FROM_GFNO, np.ndarray), \
            (f"get_feature_names_out should return numpy.ndarray, but "
             f"returned {type(RETURNED_FROM_GFNO)}")
        _ACTIVE_COLUMNS = np.array(COLUMNS)[TestCls.get_support(False)]
        assert np.array_equiv(RETURNED_FROM_GFNO, _ACTIVE_COLUMNS), \
            f"get_feature_names_out() did not return original columns"

        del junk_arg, RETURNED_FROM_GFNO, TestCls, _ACTIVE_COLUMNS

        # END ^^^ NO COLUMN NAMES PASSED (NP) ^^^

        # vvv COLUMN NAMES PASSED (PD) vvv

        TestCls = MCT(*_args, **_kwargs)
        TestCls.fit(pd.DataFrame(data=X, columns=COLUMNS), y)

        # WITH HEADER PASSED AND input_features=None, SHOULD RETURN
        # SLICED ORIGINAL COLUMNS
        _ACTIVE_COLUMNS = np.array(COLUMNS)[TestCls.get_support(False)]
        assert np.array_equiv(
            TestCls.get_feature_names_out(None),
            _ACTIVE_COLUMNS
        ), f"get_feature_names_out(None) after fit() != originally passed columns"
        del _ACTIVE_COLUMNS

        # WITH HEADER PASSED, SHOULD RAISE TypeError IF input_features
        # FOR DISALLOWED TYPES

        JUNK_COL_NAMES = [
            [*range(len(COLUMNS))], [*range(2 * len(COLUMNS))], {'a': 1, 'b': 2}
        ]
        for junk_col_names in JUNK_COL_NAMES:
            with pytest.raises(ValueError):
                TestCls.get_feature_names_out(junk_col_names)

        del JUNK_COL_NAMES

        # WITH HEADER PASSED, SHOULD RAISE ValueError IF input_features DOES
        # NOT EXACTLY MATCH ORIGINALLY FIT COLUMNS
        JUNK_COL_NAMES = \
            [np.char.upper(COLUMNS), np.hstack((COLUMNS, COLUMNS)), []]
        for junk_col_names in JUNK_COL_NAMES:
            with pytest.raises(ValueError):
                TestCls.get_feature_names_out(junk_col_names)

        # WHEN HEADER PASSED TO (partial_)fit() AND input_features IS THAT HEADER,
        # SHOULD RETURN SLICED VERSION OF THAT HEADER

        RETURNED_FROM_GFNO = TestCls.get_feature_names_out(COLUMNS)
        assert isinstance(RETURNED_FROM_GFNO, np.ndarray), \
            (f"get_feature_names_out should return numpy.ndarray, "
             f"but returned {type(RETURNED_FROM_GFNO)}")

        assert np.array_equiv(RETURNED_FROM_GFNO,
                      np.array(COLUMNS)[TestCls.get_support(False)]), \
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
            assert isinstance(__, np.ndarray), (f"get_support() did not return "
                                                f"numpy.ndarray")

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
        TestCls = MCT(*_args, **_kwargs)
        TestCls.fit(X, y)  # X IS NP ARRAY

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
        __ = np.array(COLUMNS)
        for obj_type in ['np', 'pd']:
            for diff_cols in ['more', 'less', 'same']:
                if diff_cols == 'same':
                    TEST_X = TRFM_X.copy()
                    if obj_type == 'pd':
                        TEST_X = pd.DataFrame(data=TEST_X, columns=__[TRFM_MASK])
                elif diff_cols == 'less':
                    TEST_X = TRFM_X[:, :2].copy()
                    if obj_type == 'pd':
                        TEST_X = pd.DataFrame(
                            data=TEST_X, columns=__[TRFM_MASK][:2]
                        )
                elif diff_cols == 'more':
                    TEST_X = np.hstack((TRFM_X.copy(), TRFM_X.copy()))
                    if obj_type == 'pd':
                        _COLUMNS = np.hstack((__[TRFM_MASK],
                                              np.char.upper(__[TRFM_MASK])
                        ))
                        TEST_X = pd.DataFrame(data=TEST_X, columns=_COLUMNS)

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
        del TEST_X, INV_TRFM_X

        # END inverse_transform() **********


        # ** _make_instructions()
        # ** _must_be_fitted()
        # partial_fit()
        # ** _reset()

        # set_output()
        TestCls.set_output(transform='pandas_dataframe')

        # set_params()
        TestCls.set_params(count_threshold=4)

        del TestCls

        # transform()
        # ** _validate_delete_instr()
        # ** _validate_feature_names()
        # ** _validate()

        # END ^^^ AFTER FIT ^^^ ****************************************
        # **************************************************************

    def test_access_methods_after_transform(self, X, COLUMNS, y, _args, _kwargs):

        # **************************************************************
        # vvv AFTER TRANSFORM vvv **************************************
        FittedTestCls = MCT(*_args, **_kwargs).fit(X, y)
        TransformedTestCls = MCT(*_args, **_kwargs).fit(X, y)
        TRFM_X, TRFM_Y = TransformedTestCls.transform(X, y)

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
        RETURNED_FROM_GFNO = TransformedTestCls.get_feature_names_out(COLUMNS)


        _ACTIVE_COLUMNS = np.array(COLUMNS)[TransformedTestCls.get_support(False)]
        assert np.array_equiv(RETURNED_FROM_GFNO, _ACTIVE_COLUMNS), \
            (f"get_feature_names_out() after transform did not return "
             f"sliced original columns")

        del RETURNED_FROM_GFNO
        # END ^^^ NO COLUMN NAMES PASSED (NP) ^^^

        # vvv COLUMN NAMES PASSED (PD) vvv
        PDTransformedTestCls = MCT(*_args, **_kwargs)
        PDTransformedTestCls.fit_transform(pd.DataFrame(data=X, columns=COLUMNS), y)

        # WITH HEADER PASSED AND input_features=None,
        # SHOULD RETURN SLICED ORIGINAL COLUMNS
        assert np.array_equiv(PDTransformedTestCls.get_feature_names_out(None),
                      np.array(COLUMNS)[PDTransformedTestCls.get_support(False)]), \
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


        # set_params()
        TransformedTestCls.set_params(count_threshold=4)

        # transform()
        # ** _validate_delete_instr()
        # ** _validate_feature_names()
        # ** _validate()

        del TransformedTestCls, FittedTestCls, TRFM_X, TRFM_Y

        # END ^^^ AFTER TRANSFORM ^^^ **********************************
        # **************************************************************

# END ACCESS METHODS BEFORE AND AFTER FIT AND TRANSFORM; FOR 1 RECURSION


# ACCESS ATTR BEFORE fit() AND AFTER fit_transform(), ATTR ACCURACY
# FOR 2 RECURSIONS ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **
@pytest.mark.skipif(bypass is True, reason=f"bypass")
class Test2RecursionAccessAttrsBeforeAndAfterFitAndTransform:

    def test_2_recursion_access_attrs(self, X, y, _kwargs, COLUMNS):

        NEW_X = X.copy()
        NEW_Y = y.copy()
        NEW_X_DF = pd.DataFrame(data=X, columns=COLUMNS)
        NEW_Y_DF = pd.DataFrame(data=y, columns=['y1', 'y2'])

        args = [3]
        OneRecurTestCls = MCT(*args, **_kwargs)

        _kwargs['max_recursions'] = 2

        # BEFORE FIT ***************************************************

        TwoRecurTestCls = MCT(*args, **_kwargs)

        # ALL OF THESE SHOULD GIVE AttributeError
        with pytest.raises(AttributeError):
            TwoRecurTestCls.feature_names_in_

        with pytest.raises(AttributeError):
            TwoRecurTestCls.n_features_in_

        with pytest.raises(AttributeError):
            TwoRecurTestCls.original_dtypes_

        with pytest.raises(AttributeError):
            TwoRecurTestCls.original_dtypes_ = list('abcde')

        # END BEFORE FIT ***********************************************

        # AFTER fit_transform() ****************************************
        for data_dtype in ['np', 'pd']:
            if data_dtype == 'np':
                TEST_X, TEST_Y = NEW_X.copy(), NEW_Y.copy()
            elif data_dtype == 'pd':
                TEST_X, TEST_Y = NEW_X_DF.copy(), NEW_Y_DF.copy()

            OneRecurTestCls.fit_transform(TEST_X, TEST_Y)
            TwoRecurTestCls.fit_transform(TEST_X, TEST_Y)

            assert OneRecurTestCls.n_features_in_ == X.shape[1], \
                (f"OneRecur.n_features_in_ after fit_transform() != "
                 f"number of originally passed columns")
            assert TwoRecurTestCls.n_features_in_ == X.shape[1], \
                (f"TwoRecur.n_features_in_ after fit_transform() != "
                 f"number of originally passed columns")

            # ONLY EXCEPTION SHOULD BE feature_names_in_ WHEN NUMPY
            if data_dtype == 'pd':
                assert np.array_equiv(TwoRecurTestCls.feature_names_in_, COLUMNS), \
                    (f"2 recurrence feature_names_in_ after fit_transform() != "
                     f"originally passed columns")

                assert np.array_equiv(TwoRecurTestCls.feature_names_in_,
                    OneRecurTestCls.feature_names_in_), \
                    (f"2 recurrence feature_names_in_ after fit_transform() != 1 "
                     f"recurrence feature_names_in_ after fit_transform()")
            elif data_dtype == 'np':
                with pytest.raises(AttributeError):
                    TwoRecurTestCls.feature_names_in_

            # n_features_in_ SHOULD BE EQUAL FOR OneRecurTestCls AND TwoRecurTestCls
            _, __ = OneRecurTestCls.n_features_in_, TwoRecurTestCls.n_features_in_
            assert _ == __, (f"OneRecurTestCls.n_features_in_ ({_}) != "
                             f"TwoRecurTestcls.n_features_in_ ({__})")
            del _, __

            assert np.array_equiv(
                TwoRecurTestCls._original_dtypes,
                ['int' for _ in range(X.shape[1])]
            ), f"_original_dtypes after fit_transform() != originally passed dtypes"

        # END AFTER fit_transform() ************************************

        del NEW_X, NEW_Y, NEW_X_DF, NEW_Y_DF, data_dtype, TEST_X, TEST_Y
        del OneRecurTestCls, TwoRecurTestCls

# END ACCESS ATTR BEFORE fit() AND AFTER fit_transform()
# ATTR ACCURACY; FOR 2 RECURSIONS **************************************


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

        return MCT(*args, **_kwargs)


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

        return MCT(*args, **_kwargs)

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
        TwoRecurTestCls.set_params(count_threshold=4)

        del TwoRecurTestCls

        TwoRecurTestCls = MCT(*_args, **_kwargs)
        # transform()
        with pytest.raises(AttributeError):
            TwoRecurTestCls.transform(X, y)

        # ** _validate_delete_instr()
        # ** _validate_feature_names()
        # ** _validate()

        # END ^^^ BEFORE fit_transform() ^^^ ***************************
        # **************************************************************

        del TwoRecurTestCls


    def test_after_fit_transform(self, OneRecurTestCls, TwoRecurTestCls, X,
        COLUMNS, y, _args, _kwargs):

        ONE_RCR_TRFM_X, ONE_RCR_TRFM_Y = OneRecurTestCls.fit_transform(X, y)
        TWO_RCR_TRFM_X, TWO_RCR_TRFM_Y = TwoRecurTestCls.fit_transform(X, y)

        # **************************************************************
        # vvv AFTER fit_transform() vvv ********************************

        # ** _base_fit()
        # ** _check_is_fitted()

        # ** test_threshold()
        assert not np.array_equiv(ONE_RCR_TRFM_X, TWO_RCR_TRFM_X), \
            f"ONE_RCR_TRFM_X == TWO_RCR_TRFM_X when it shouldnt"

        assert (OneRecurTestCls._total_counts_by_column !=
                TwoRecurTestCls._total_counts_by_column), \
            (f"OneRecurTestCls._total_counts_by_column == "
             f"TwoRecurTestCls._total_counts_by_column when it shouldnt")

        _ONE_delete_instr = OneRecurTestCls._make_instructions(_args[0])
        _TWO_delete_instr = TwoRecurTestCls._make_instructions(_args[0])
        # THE FOLLOWING MUST BE TRUE BECAUSE TEST DATA BUILD VALIDATION
        # REQUIRES 2 RECURSIONS W CERTAIN KWARGS DOES DELETE SOMETHING
        assert _TWO_delete_instr != _ONE_delete_instr, \
            (f"fit-trfmed 2 recursion delete instr == fit-trfmed 1 recursion "
             f"delete instr and should not")

        # THE NUMBER OF COLUMNS IN BOTH delete_instr DICTS ARE EQUAL
        assert len(_TWO_delete_instr) == len(_ONE_delete_instr), \
            (f"number of columns in TwoRecurTestCls delete instr != number of "
             f"columns in OneRecurTestCls delete instr")

        # LEN OF INSTRUCTIONS IN EACH COLUMN FOR TWO RECUR MUST BE >=
        # INSTRUCTIONS FOR ONE RECUR BECAUSE THEYVE BEEN MELDED
        for col_idx in _ONE_delete_instr:
            _, __ = len(_TWO_delete_instr[col_idx]), len(_ONE_delete_instr[col_idx])
            assert _ >= __, (f"number of instruction in TwoRecurTestCls count "
                         f"is not >= number of instruction in OneRecurTestCls"
            )

        # ALL THE ENTRIES FROM 1 RECURSION ARE IN THE MELDED INSTRUCTION DICT
        # OUTPUT OF MULTIPLE RECURSIONS
        for col_idx in _ONE_delete_instr:
            for unq in list(map(str, _ONE_delete_instr[col_idx])):
                if unq in ['INACTIVE', 'DELETE COLUMN']:
                    continue
                assert unq in list(map(str, _TWO_delete_instr[col_idx])), \
                    f"{unq} is in 1 recur delete instructions but not 2 recur"

        del _ONE_delete_instr, _TWO_delete_instr, _, __, col_idx, unq

        TwoRecurTestCls.test_threshold(clean_printout=True)
        print(f'^^^ mask building instructions should be displayed above ^^^')

        with pytest.raises(ValueError):
            TwoRecurTestCls.test_threshold(2 * _args[0])

        # fit()
        # fit_transform()

        # get_feature_names_out() **************************************
        # vvv NO COLUMN NAMES PASSED (NP) vvv

        # WITH NO HEADER PASSED AND input_features=None, SHOULD RETURN
        # SLICED ['x0', ..., 'x(n-1)]
        _COLUMNS = np.array([f"x{i}" for i in range(len(COLUMNS))])
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
            _COLUMNS = [f"x{i}" for i in range(2 * len(COLUMNS))]
            TwoRecurTestCls.get_feature_names_out(_COLUMNS)
            del _COLUMNS

        # WHEN NO HEADER PASSED TO fit_transform() AND VALID input_features,
        # SHOULD RETURN SLICED PASSED COLUMNS
        RETURNED_FROM_GFNO = TwoRecurTestCls.get_feature_names_out(COLUMNS)
        assert isinstance(RETURNED_FROM_GFNO, np.ndarray), \
            (f"TwoRecur.get_feature_names_out should return numpy.ndarray, "
             f"but returned {type(RETURNED_FROM_GFNO)}")

        assert np.array_equiv(RETURNED_FROM_GFNO,
            np.array(COLUMNS)[TwoRecurTestCls.get_support(False)]), \
            f"TwoRecur.get_feature_names_out() did not return original columns"

        del RETURNED_FROM_GFNO

        # END ^^^ NO COLUMN NAMES PASSED (NP) ^^^

        # vvv COLUMN NAMES PASSED (PD) vvv
        ONE_RCR_TRFM_X, ONE_RCR_TRFM_Y = \
            OneRecurTestCls.fit_transform(
                pd.DataFrame(data=X, columns=COLUMNS), y
            )

        TWO_RCR_TRFM_X, TWO_RCR_TRFM_Y = \
            TwoRecurTestCls.fit_transform(
                pd.DataFrame(data=X, columns=COLUMNS), y
            )

        # WITH HEADER PASSED AND input_features=None:
        # SHOULD RETURN SLICED ORIGINAL COLUMNS
        assert np.array_equiv(
            TwoRecurTestCls.get_feature_names_out(None),
            np.array(COLUMNS)[TwoRecurTestCls.get_support(False)]
            ), (f"TwoRecur.get_feature_names_out(None) after fit_transform() != "
            f"sliced originally passed columns"
        )

        # WHEN HEADER PASSED TO fit_transform() AND input_features IS THAT HEADER,
        # SHOULD RETURN SLICED VERSION OF THAT HEADER
        RETURNED_FROM_GFNO = TwoRecurTestCls.get_feature_names_out(COLUMNS)
        assert isinstance(RETURNED_FROM_GFNO, np.ndarray), \
            (f"get_feature_names_out should return numpy.ndarray, but returned "
             f"{type(RETURNED_FROM_GFNO)}")
        assert np.array_equiv(
            RETURNED_FROM_GFNO,
            np.array(COLUMNS)[TwoRecurTestCls.get_support(False)]
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
                assert len(_ONE) == X.shape[0], \
                    (f"row_support vector length for 1 recursion != rows in "
                     f"passed data"
                )
                assert len(_TWO) == X.shape[0], \
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
                assert len(_) == X.shape[1], \
                    f"1 recursion len(get_support({_indices})) != X columns"
                assert len(__) == X.shape[1], \
                    f"2 recursion len(get_support({_indices})) != X columns"
                # NUM COLUMNS IN 1 RECURSION MUST BE <= NUM COLUMNS IN X
                assert sum(_) <= X.shape[1], \
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
                assert len(_) <= X.shape[1], \
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
        TwoRecurTestCls = MCT(count_threshold=3, **_kwargs)
        # X IS NP ARRAY
        TRFM_X, TRFM_Y = TwoRecurTestCls.fit_transform(X, y)

        # SHOULD RAISE ValueError WHEN COLUMNS DO NOT EQUAL NUMBER OF RETAINED COLUMNS
        __ = np.array(COLUMNS)
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

        # set_params()
        TwoRecurTestCls.set_params(count_threshold=7)


        # transform()
        # ** _validate_delete_instr()
        # ** _validate_feature_names()
        # ** _validate()

        del OneRecurTestCls, TwoRecurTestCls
        # END ^^^ AFTER fit_transform() ^^^ ****************************
        # **************************************************************


# END ACCESS METHODS BEFORE AND AFTER FIT AND TRANSFORM; FOR 2 RECURSIONS






