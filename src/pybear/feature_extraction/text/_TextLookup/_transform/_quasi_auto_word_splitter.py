# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from copy import deepcopy

from .....utilities._view_text_snippet import _view_text_snippet
from .....data_validation import validate_user_input as vui



def _quasi_auto_word_splitter(
    _word_idx: int,
    _line: list[str],
    _KNOWN_WORDS: list[str],
    _verbose: bool
) -> list[str]:

    """
    Look if the current 'word' that is not in the Lexicon is actually an
    erroneous compounding of two valid words that are in the Lexicon.
    Working from left to right in the word, starting after the second
    letter and stopping before the second-to-last letter, look for the
    first valid split comprised of 2 halves each with 2 or more
    characters. Prompt the user if they want to keep the proposed split.
    If not, continue looking for and proposing valid splits until the
    user accepts a split or all valid splits are exhausted.


    Parameters
    ----------
    _word_idx:
        int - the index of the active word in its line.
    _line:
        list[str] - the full line that the active word is in.
    _KNOWN_WORDS:
        list[str] - All the words in the Lexicon and any words that have
        been put into LEXICON_ADDENDUM in the current session.
    _verbose:
        bool - whether to display helpful information.


    Returns
    -------
    _NEW_LINE:
        list[str] - If a valid split is found and the user accepts the
        split, a copy of _line is made, the old word is removed from the
        copy, the word is split, and the two new words are inserted into
        the copy starting at the position of the original word. In a
        nutshell, if no split is found, an empty list is returned. If a
        split is found, modify the line with the new words and return
        the modified line.


    """


    _word = _line[_word_idx]

    _NEW_LINE = []
    for split_idx in range(2, len(_word) - 1):
        if _word[:split_idx] in _KNOWN_WORDS and _word[split_idx:] in _KNOWN_WORDS:
            print(_view_text_snippet(_line, _word_idx, _span=9))
            print(f"\n*{_word}* IS NOT IN LEXICON\n")
            print(f'\n*** RECOMMEND *{_word[:split_idx]}* AND '
                  f'*{_word[split_idx:]}* ***\n')
            # if user does not like the suggested split, continue making
            # & recommending splits. if no more splits, return the empty _NEW_LINE.
            if vui.validate_user_str(f'Accept? (y/n) > ', 'YN') == 'Y':

                _NEW_LINE = deepcopy(_line)
                # insert backwards
                _NEW_LINE.pop(_word_idx)
                _NEW_LINE.insert(_word_idx, _word[split_idx:])
                _NEW_LINE.insert(_word_idx, _word[:split_idx])

                if _verbose:
                    print(
                        f'\n*** SUBSTITUTING *{_word}* WITH *{_word[:split_idx]}* '
                          f'AND *{_word[split_idx:]}*\n'
                    )

                break


    return _NEW_LINE




