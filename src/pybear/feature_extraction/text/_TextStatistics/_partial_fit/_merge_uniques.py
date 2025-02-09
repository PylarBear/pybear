# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Sequence

from .._validation._uniques import _val_uniques


# pizza this may not be needed, _word_frequency is getting uniques also

def _merge_uniques(
    _current_uniques: Sequence[str],
    _uniques: Sequence[str]
) -> Sequence[str]:


    _val_uniques(_current_uniques)
    _val_uniques(_uniques)

    # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    _uniques = set(_uniques).union(set(_current_uniques))


    return _uniques


