# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



def _display_lexicon_update(
    LEXICON_ADDENDUM: list[str]
    ) -> None:

    """
    Prints contents of LEXICON_ADDENDUM object for copy and paste into
    LEXICON.

    Parameters
    ----------
    LEXICON_ADDENDUM:
        list[str] - list of words to be manually added to the Lexicon

    Return
    ------
    -
        None

    """

    if len(LEXICON_ADDENDUM) == 0:
        print(f'\n *** LEXICON ADDENDUM IS EMPTY *** \n')

    else:
        LEXICON_ADDENDUM.sort()
        print(f'\n *** Copy and paste these words into Lexicon:\n')
        print(f'[')
        for _ in LEXICON_ADDENDUM:
            print(f'   "{_}"{"" if _ == LEXICON_ADDENDUM[-1] else ","}')
        print(f']')
        print()

        input(f'\n*** Paused to allow copy, hit Enter to continue > ')

    return




























