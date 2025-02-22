# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import numbers

from typing_extensions import Union
import numpy.typing as npt

import numpy as np





def justify(
    _X_WIP: Union[list[str], list[list[str]], npt.NDArray[str]],
    _chars: Union[numbers.Integral, None]
) -> None:

    """
    Fit text as strings or as lists to user-specified number of
    characters per row.


    Parameters
    ----------
    _X_WIP:

    _chars:
        int - number of characters per row


    Return
    ------


    """

    # ALSO SEE text.notepad_justifier FOR SIMILAR CODE, IF EVER CONSOLIDATING

    # CONVERT TO LIST OF LISTS
    converted = False
    if not self.is_list_of_lists:
        self.as_list_of_lists()
        converted = True

    if not _chars is None:
        arg_kwarg_validater(
            _chars,
            'characters',
            list(range(30, 50001)),
            'TC',
            'justify'
        )
    elif _chars is None:
        # DONT PUT THIS IN akv(return_if_none=)... PROMPTS USER FOR
        # sINPUT BEFORE ENDING args/kwargs TO akv
        _chars = vui.validate_user_int(
            f'\nEnter number of characters per line (min=30, max=50000) > ',
            min=30,
            max=50000
        )

    seed = f''
    del _chars
    NEW_TXT = [[]]
    for row_idx in range(len(_X_WIP)):
        for word_idx in range(len(_X_WIP[row_idx])):
            new_word = _X_WIP[row_idx][word_idx].strip()
            if (len(seed) + len(new_word)) <= _chars:
                NEW_TXT[-1].append(new_word)
                seed += new_word + ' '
            else:
                NEW_TXT.append([new_word])
                seed = f'{new_word} '
    else:
        if len(seed) > 0:
            NEW_TXT.append(seed.split())

    del _chars, seed, new_word

    _X_WIP = NEW_TXT
    del NEW_TXT
    self.is_list_of_lists = False

    # OBJECT WAS WORKED ON AS LIST OF LISTS, BUT OUTPUT IS LIST OF STRS
    if converted:
        # MEANING THAT IS WAS list_of_strs TO START WITH, JUST LEAVE AS IS
        pass
    elif not converted:
        # OTHERWISE WAS LIST OF LISTS TO START, SO CONVERT BACK TO LIST OF LISTS
        self.as_list_of_lists()
        map(str.strip, _X_WIP)
    del converted


