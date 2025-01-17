# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#





import pytest



pytest.skip(reason=f"pizza finish this!", allow_module_level=True)



class TestGetFeatureNamesOut:


    # AFTER FIT #################################################################
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
    TestCls.fit(pd.DataFrame(data=_X_np, columns=_columns), y)

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

    # END AFTER FIT #################################################################


    # AFTER TRANSFORM ###############################################################
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
        pd.DataFrame(data=_X_np, columns=_columns), y
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
    # END AFTER TRANSFORM ###############################################################





















