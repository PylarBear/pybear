# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Callable
from typing_extensions import Union



def _val_ngcallable(
    _ngcallable: Union[Callable[[list[str]], str], None]
) -> None:
    """Validate the ngram callable.

    Must be a callable or None. The output and signature of the callable
    are not validated.

    Parameters
    ----------
    _ngcallable : Union[Callable[[list[str]], str], None]
        The callable applied to ngram sequences to produce a contiguous
        string sequence.

    Returns
    -------
    None

    """


    if _ngcallable is None:
        return

    if not callable(_ngcallable):
        raise TypeError(f"'ngcallable' must be a callable or None.")





