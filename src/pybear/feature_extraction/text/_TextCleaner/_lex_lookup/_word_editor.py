# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from pybear.data_validation import validate_user_input as vui



def _word_editor(prompt:str) -> str:

    """
    Prompt for single words entered by user and validate.

    Parameters
    ----------
    prompt:
        str - the prompt to display when asking user for a word

    Return
    ------
    -
        word: str - the validated word entered by user


    """

    if not isinstance(prompt, str):
        raise TypeError(f"'prompt', if passed, must be a string, otherwise None")


    while True:
        word = input(f'{prompt} > ')
        if vui.validate_user_str(
                f'USER ENTERED *{word}* -- ACCEPT? (Y/N) > ', 'YN') == 'Y':
            break

    return word
























