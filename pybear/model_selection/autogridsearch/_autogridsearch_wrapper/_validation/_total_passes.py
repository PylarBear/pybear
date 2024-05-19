# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



def _total_passes(_total_passes: int) -> int:

    """
    Validate _total_passes --- require an integer in range [0, 100]

    """

    err_msg = f'total_passes must be an integer in range [1,100]'

    if 'INT' not in str(type(_total_passes)).upper():
        raise TypeError(err_msg)

    if _total_passes not in range(1, 101):

        raise ValueError(err_msg)


    return _total_passes





