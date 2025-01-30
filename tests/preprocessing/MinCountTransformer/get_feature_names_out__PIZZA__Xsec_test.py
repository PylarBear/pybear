# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



# pizza placeholder

"""
# 1 RCR after transform()

        # get_feature_names_out() **************************************
        # vvv NO COLUMN NAMES PASSED (NP) vvv

        # # WHEN NO HEADER PASSED TO (partial_)fit() AND VALID input_features,
        # SHOULD RETURN ORIGINAL (UNSLICED) _columns
        RETURNED_FROM_GFNO = TransformedTestCls.get_feature_names_out(_columns)


        _ACTIVE__columns = np.array(_columns)[TransformedTestCls.get_support(False)]
        assert np.array_equiv(RETURNED_FROM_GFNO, _ACTIVE__columns), \
            (f"get_feature_names_out() after transform did not return "
             f"sliced original columns")

        del RETURNED_FROM_GFNO
        # END ^^^ NO COLUMN NAMES PASSED (NP) ^^^

        # vvv COLUMN NAMES PASSED (PD) vvv
        PDTransformedTestCls = MCT(*_args, **_kwargs)
        PDTransformedTestCls.fit_transform(pd.DataFrame(data=X, columns=_columns), y)

        # WITH HEADER PASSED AND input_features=None,
        # SHOULD RETURN SLICED ORIGINAL _columns
        assert np.array_equiv(PDTransformedTestCls.get_feature_names_out(None),
                      np.array(_columns)[PDTransformedTestCls.get_support(False)]), \
            (f"get_feature_names_out(None) after transform() != "
             f"originally passed columns")

        del PDTransformedTestCls
        # END ^^^ COLUMN NAMES PASSED (PD) ^^^

        # END get_feature_names_out() **********************************


"""



"""
    1 RCR after fit()

        # get_feature_names_out() **************************************
        # vvv NO COLUMN NAMES PASSED (NP) vvv
        # **** CAN ONLY TAKE LIST-TYPE OF STRS OR None
        JUNK_ARGS = [float('inf'), np.pi, 'garbage', {'junk': 3},
                     [*range(len(_columns))]
        ]

        for junk_arg in JUNK_ARGS:
            with pytest.raises(ValueError):
                TestCls.get_feature_names_out(junk_arg)

        del JUNK_ARGS

        # WITH NO HEADER PASSED AND input_features=None, SHOULD RETURN
        # ['x0', ..., 'x(n-1)][COLUMN MASK]
        __columns = np.array([f"x{i}" for i in range(len(_columns))])
        ACTIVE__columns = __columns[TestCls.get_support(False)]
        del __columns
        assert np.array_equiv(
            TestCls.get_feature_names_out(None),
            ACTIVE__columns
        ), (f"get_feature_names_out(None) after fit() != sliced array of "
            f"generic headers")
        del ACTIVE__columns

        # WITH NO HEADER PASSED, SHOULD RAISE ValueError IF
        # len(input_features) != n_features_in_
        with pytest.raises(ValueError):
            TestCls.get_feature_names_out(
                [f"x{i}" for i in range(2 * len(_columns))]
            )

        # WHEN NO HEADER PASSED TO (partial_)fit() AND VALID input_features,
        # SHOULD RETURN SLICED PASSED _columns
        RETURNED_FROM_GFNO = TestCls.get_feature_names_out(_columns)
        assert isinstance(RETURNED_FROM_GFNO, np.ndarray), \
            (f"get_feature_names_out should return numpy.ndarray, but "
             f"returned {type(RETURNED_FROM_GFNO)}")
        _ACTIVE__columns = np.array(_columns)[TestCls.get_support(False)]
        assert np.array_equiv(RETURNED_FROM_GFNO, _ACTIVE__columns), \
            f"get_feature_names_out() did not return original columns"

        del junk_arg, RETURNED_FROM_GFNO, TestCls, _ACTIVE__columns

        # END ^^^ NO COLUMN NAMES PASSED (NP) ^^^

        # vvv COLUMN NAMES PASSED (PD) vvv

        TestCls = MCT(*_args, **_kwargs)
        TestCls.fit(pd.DataFrame(data=X, columns=_columns), y)

        # WITH HEADER PASSED AND input_features=None, SHOULD RETURN
        # SLICED ORIGINAL _columns
        _ACTIVE__columns = np.array(_columns)[TestCls.get_support(False)]
        assert np.array_equiv(
            TestCls.get_feature_names_out(None),
            _ACTIVE__columns
        ), f"get_feature_names_out(None) after fit() != originally passed columns"
        del _ACTIVE__columns

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
        # NOT EXACTLY MATCH ORIGINALLY FIT _columns
        JUNK_COL_NAMES = \
            [np.char.upper(_columns), np.hstack((_columns, _columns)), []]
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

"""


"""
2 RCR after fit_transform

        # get_feature_names_out() **************************************
        # vvv NO COLUMN NAMES PASSED (NP) vvv

        # WITH NO HEADER PASSED AND input_features=None, SHOULD RETURN
        # SLICED ['x0', ..., 'x(n-1)]
        __columns = np.array([f"x{i}" for i in range(len(_columns))])
        _ACTIVE__columns = __columns[TwoRecurTestCls.get_support(False)]
        del __columns
        assert np.array_equiv(
            TwoRecurTestCls.get_feature_names_out(None),
            _ACTIVE__columns
        ), (f"get_feature_names_out(None) after fit_transform() != sliced "
            f"array of generic headers"
        )
        del _ACTIVE__columns

        # WITH NO HEADER PASSED, SHOULD RAISE ValueError IF len(input_features) !=
        # n_features_in_
        with pytest.raises(ValueError):
            __columns = [f"x{i}" for i in range(2 * len(_columns))]
            TwoRecurTestCls.get_feature_names_out(__columns)
            del __columns

        # WHEN NO HEADER PASSED TO fit_transform() AND VALID input_features,
        # SHOULD RETURN SLICED PASSED _columns
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
            OneRecurTestCls.fit_transform(
                pd.DataFrame(data=X, columns=_columns), y
            )

        TWO_RCR_TRFM_X, TWO_RCR_TRFM_Y = \
            TwoRecurTestCls.fit_transform(
                pd.DataFrame(data=X, columns=_columns), y
            )

        # WITH HEADER PASSED AND input_features=None:
        # SHOULD RETURN SLICED ORIGINAL _columns
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


"""



