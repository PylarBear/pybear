# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from .._type_aliases import TotalCountsByColumnType

import numpy as np




def _tcbc_update(
        old_tcbc: TotalCountsByColumnType, # use deepcopy(self._total_counts_by_column)
        recursion_tcbc: TotalCountsByColumnType, # use RecursiveCls._total_counts_by_column
        MAP_DICT: dict[int: int]
    ) -> TotalCountsByColumnType:

    """
    Iterate over RecursiveCls._total_counts_by_column (recursion_tcbc)
    and compare the counts to the corresponding ones in self._tcbc
    (old_tcbc). If RecursiveCls's value is lower, put it into self's.
    _total_counts_by_column is a dict of unqs and cts for each column in
    X. If X had features deleted, must map feature locations in
    RecursiveCls._tcbc to their old locations in self._tcbc.

    Parameters
    ----------
    old_tcbc: TotalCountsByColumnType, # use deepcopy(self._total_counts_by_column)
    recursion_tcbc: TotalCountsByColumnType, # use RecursiveCls._total_counts_by_column
    MAP_DICT: dict[int: int] - dictionary mapping a feature's location in
        Recursion._tcbc to its (possibly different) location in self._tcbc.

    Return
    ------
    old_tcbc:
        TotalCountsByColumnType - updated with counts from Recursion._tcbc

    """

    for new_col_idx in recursion_tcbc:

        old_col_idx = MAP_DICT[new_col_idx]

        for unq, ct in recursion_tcbc[new_col_idx].items():

            OLD_TCBC_UNIQUES = np.char.lower(
                np.fromiter(old_tcbc[old_col_idx].keys(), dtype='<U100')
            )
            # if not in at first look, look again after convert to str
            if unq not in old_tcbc[old_col_idx] and \
                    str(unq).lower() not in OLD_TCBC_UNIQUES:
                raise AssertionError(f"algorithm failure, unique in a "
                    f"deeper recursion is not in the previous recursion"
                )

            del OLD_TCBC_UNIQUES

            # there is a problem of matching that largely seems to impact
            # nan. look to match the recursion_tcbc unq to a unq in
            # old_tcbc. if no match convert recursion_tcbc unq to str &
            # try again against old_tcbc, if still no match convert
            # old_tcbc keys to str and try recursion_tcbc as str again,
            # if still no match raise exception.
            while True:
                try:
                    __ = old_tcbc[old_col_idx][unq]
                    break
                except:
                    pass

                try:
                    __ = old_tcbc[old_col_idx][str(unq).lower()]
                    break
                except:
                    pass

                try:
                    __ = dict((
                        zip(list(map(str, old_tcbc[old_col_idx].keys())),
                        list(old_tcbc[old_col_idx].values()))
                    ))['nan']
                    break
                except:
                    raise ValueError(f"could not access key {unq} in "
                        f"_total_counts_by_column"
                    )

            assert ct <= __, (f"algorithm failure, count of a unique "
                f"in a deeper recursion is > the count of the same unique "
                f"in a higher recursion, can only be <="
            )

            # all of that just to get this one number __
            if ct < __:
                old_tcbc[old_col_idx][unq] = ct


    del new_col_idx, old_col_idx, unq, ct, __

    return old_tcbc









