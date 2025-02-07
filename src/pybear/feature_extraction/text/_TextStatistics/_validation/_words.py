# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Sequence



def _val_words(words: Sequence[str]) -> None:

    """
    Validate 'words'.

    - Must be 1D sequence of strings
    - cannot be empty
    - all strings must be under 30 characters


    Parameters
    ----------
    words:
        Sequence[str] - a single list-like vector of words to report
        statistics for. Words do not need to be in the Lexicon.


    Return
    ------
    -
        None


    """



    err_msg = (f"'words' must be passed as a list-like vector of strings, "
               f"cannot be empty")

    try:
        iter(words)
        if isinstance(words, (dict, str)):
            raise Exception
    except:
        raise TypeError(err_msg)

    if len(words) == 0:
        raise ValueError(err_msg)

    if not all(map(isinstance, words, (str for _ in words))):
        raise TypeError(err_msg)

    del err_msg

    if max(map(len, words)) > 30:
        raise ValueError(
            f"'words' is likely not a vector of individual words. "
            f"\npass 'words' as a list-like of individual words only."
        )




