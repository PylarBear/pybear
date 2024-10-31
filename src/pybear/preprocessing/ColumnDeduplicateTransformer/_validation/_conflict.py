# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from typing import Literal




def _val_conflict(
    _conflict: Literal['raise', 'ignore']
) -> None:

    """
    Validate 'conflict' is either 'raise' or 'ignore'.


    Parameters
    ----------
    _conflict:
        Literal['raise', 'ignore'] - Ignored when :param: do_not_drop is
        not passed. Instructs CDT how to deal with a conflict between
        the instructions in :param: keep and :param: do_not_drop. A
        conflict arises when the instruction in :param: keep ('first',
        'last', 'random') is applied and column in :param: do_not_drop
        is found to be a member of the columns to be deleted. When :param:
        conflict is 'raise', an exception is raised in the case of such
        a conflict. When :param: conflict is 'ignore', there are 2
        possible scenarios:

        1) when only one column in :param: do_not_drop is among the
        columns to be deleted, the :param: keep instruction is overruled
        and the do_not_drop column is kept

        2) when multiple columns in :param: do_not_drop are among the
        columns to be deleted, the :param: keep instruction ('first',
        'last', 'random') is applied to the set of do-not-delete columns
        that would be marked for deletion --- this may not give the same
        result as applying the :param: keep instruction  to the entire
        set of duplicate columns. This also causes at least one member
        of the columns not to be dropped to be deleted.


    Return
    ------
    -
        None




    """


    # must be 'raise' or 'ignore'

    err_msg = f"'conflict' must be literal 'raise' or 'ignore'"

    if not isinstance(_conflict, str):
        raise TypeError(err_msg)

    if _conflict.lower() not in ['raise', 'ignore']:
        raise ValueError(err_msg)





