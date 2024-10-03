# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

from .._type_aliases import KeepType



def _val_keep(_keep:KeepType) -> None:

    _err_msg = lambda _required: f"'keep' must be one of {', '.join(_required)}"
    _required = ('first', 'last', 'random')

    if not isinstance(_keep, str):
        raise TypeError(_err_msg(_required))

    if sum([_ == _keep for _ in _required]) != 1:
        raise ValueError(_err_msg(_required))
    del _err_msg, _required












