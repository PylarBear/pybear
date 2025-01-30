# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#




# pizza finish this!


"""
# 1 RCR after transform (the only place possible)


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




"""



"""
2 RCR after fit_transform()


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


"""


