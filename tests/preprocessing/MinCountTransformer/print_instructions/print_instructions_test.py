# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


# pizza this is for testing the MCT print_instructions method directly.
# make a decision what u want to do here.


"""
from attr_method_access, 2 RCR after transform.

        # ** print_instructions()
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

        # THE NUMBER OF _columns IN BOTH delete_instr DICTS ARE EQUAL
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
"""






