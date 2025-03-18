# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from ......data_validation import validate_user_input as vui


def _word_editor(
    _word: str,
    _prompt: str
) -> str:

    """
    Validation function for single words entered by user.


    Parameter
    ---------
    _word:
        str - the word prompting a new entry by the user.
    _prompt:
        str - a special prompt.


    Return
    ------
    -
        _word: str - the new word entered by the user.


    """


    if not isinstance(_word, str):
        raise TypeError(f"'word' must be a string")

    if not isinstance(_prompt, str):
        raise TypeError(f"'prompt' must be a string")

    while True:

        _word = input(f'{_prompt} > ').upper()

        if vui.validate_user_str(
            f'User entered *{_word}* -- accept? (y/n) > ', 'YN'
        ) == 'Y':
            break


    return _word


