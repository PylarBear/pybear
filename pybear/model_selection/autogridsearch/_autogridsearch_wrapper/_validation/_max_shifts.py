# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



def _max_shifts(_max_shifts:int) -> int:

    """
    Validate _max_shifts --- require an integer in range [0, 100]

    """


    err_msg = f'total_passes must be an integer in range [1,100]'

    if 'INT' not in str(type(_max_shifts)).upper():
        raise TypeError(err_msg)

    if _max_shifts not in range(1, 101):

        raise ValueError(err_msg)


    return _max_shifts
