# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from .....data_validation import validate_user_input as vui

from ._view_snippet import _view_snippet
from ._word_editor import _word_editor



def _manual_word_splitter(
    _word: str,
    _word_idx: int,
    _line: list[str],
    _KNOWN_WORDS: list[str],
    _verbose
) -> list[str]:

    """
    The user has opted to enter into this functionality. The word may
    not otherwise be recognized as a candidate to split. Prompt the user
    for the number of splits. For that number of times, prompt the user
    for words that will replace the single word in the line.


    Parameters
    ----------
    _word:
        str - the current word being processed by TextLookup. We know
        its not in the Lexicon.
    _word_idx:
        int - the index of '_word' in its line.
    _line:
        list[str] - the line that '_word' is in.
    _KNOWN_WORDS:
        list[str] - All the words in the Lexicon and any words that have
        been put into LEXICON_ADDENDUM in the current session.
    _verbose:
        bool - whether to display helpful information.


    Returns
    -------
    _NEW_WORDS:
        list[str] - The words the user has entered and acknowledged as
        the new words to replace the single word.


    """


    while True:

        # ask user number of new words to make -- -- -- -- -- -- -- -- --
        _prompt = (
            f'Enter number of ways to split  *{_word}*  '
            f'in  *{_view_snippet(_line, _word_idx)}* > '
        )
        new_word_ct = vui.validate_user_int(_prompt, min=1, max=100)
        del _prompt
        # END ask user number of new words to make -- -- -- -- -- -- --

        # ask user for the new words -- -- -- -- -- -- -- -- -- -- -- --
        _NEW_WORDS = ['' for _ in range(new_word_ct)]
        for slot_idx in range(int(new_word_ct)):
            _prompt = (
                f'Enter word for slot {slot_idx + 1} '
                f'(of {new_word_ct}) replacing  *{_word}*  '
                f'in  *{_view_snippet(_line, _word_idx)}*'
            )
            _NEW_WORDS[slot_idx] = _word_editor(_line[_word_idx], _prompt=_prompt)
        del _prompt
        # END ask user for the new words -- -- -- -- -- -- -- -- -- -- --

        # ask user if the new words are correct -- -- -- -- -- -- -- --
        _prompt = f'User entered *{", ".join(_NEW_WORDS)}* > accept? (y/n) > '
        if vui.validate_user_str(_prompt, 'YN') == 'Y':
            del _prompt
            break
        else:
            continue
        # END ask user if the new words are correct -- -- -- -- -- -- --


    if _verbose:
        print(f'\n*** SUBSTITUTING "{_word}" WITH "{", ".join(_NEW_WORDS)}"\n')


    return _NEW_WORDS






