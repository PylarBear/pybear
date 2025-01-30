# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


# pizza finish this!



"""

    1 RCR after transform? or fit?

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
"""


"""

    2 RCR after fit_transform()

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
                # MUST EQUAL NUMBER OF _columns IN X
                assert len(_) == X.shape[1], \
                    f"1 recursion len(get_support({_indices})) != X columns"
                assert len(__) == X.shape[1], \
                    f"2 recursion len(get_support({_indices})) != X columns"
                # NUM _columns IN 1 RECURSION MUST BE <= NUM _columns IN X
                assert sum(_) <= X.shape[1], \
                    (f"impossibly, number of columns kept by 1 recursion > number "
                     f"of columns in X")
                # NUM _columns IN 2 RECURSION MUST BE <= NUM _columns IN 1 RECURSION
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
                # ONE RECURSION _columns MUST BE <= n_features_in_
                assert len(_) <= X.shape[1], \
                    f"impossibly, 1 recursion len(get_support({_indices})) > X columns"
                # TWO RECURSION _columns MUST BE <= ONE RECURSION _columns
                assert len(__) <= len(_), \
                    (f"2 recursion len(get_support({_indices})) > 1 "
                     f"recursion len(get_support({_indices}))")
                # INDICES IN TWO RECUR MUST ALSO BE IN ONE RECUR
                for col_idx in __:
                    assert col_idx in _, (f"Columns that are to be kept by "
                              f"2nd recur were not kept by 1st recur")

        del TwoRecurTestCls, _, __, _indices, col_idx


"""





















