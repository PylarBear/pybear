# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from typing import Union



def _val_verbose(
        _verbose: Union[bool, int, float]
    ) -> int:


    err_msg = f"verbose must be a bool or a numeric > 0"

    try:
        if isinstance(_verbose, bool):
            if _verbose is True:
                _verbose = 10
            elif _verbose is False:
                _verbose = 0
        float(_verbose)
        _verbose = int(round(_verbose, 0))
    except:
        raise TypeError(err_msg)

    if _verbose < 0:
        raise ValueError(err_msg)

    del err_msg

    return _verbose



