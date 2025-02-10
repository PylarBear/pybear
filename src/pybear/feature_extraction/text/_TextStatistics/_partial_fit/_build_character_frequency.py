# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import numbers

from .._validation._word_frequency import _val_word_frequency



def _build_character_frequency(
    _word_frequency: dict[str, numbers.Integral]
) -> dict[str, numbers.Integral]:

    """
    Build a dictionary that contains all the unique characters in
    '_word_frequency' as keys and the number of times that that character
    appears as the values.


    Parameters
    ----------
    _word_frequency:
        dict[str, numbers.Integral] - the dictionary holding the unique
        strings passed to the current partial fit and their respective
        frequencies.


    Return
    ------
    -
        _character_frequency: dict[str, numbers.Integral] - a dictionary
        that holds the unique characters passed to this partial fit and
        their respective number of appearances as values.


    """

    _val_word_frequency(_word_frequency)

    # pizza, maybe do some benchmarking on this.

    _character_frequency: dict[str: numbers.Integral] = {}

    for _string, _ct in _word_frequency.items():
        for _char in str(_string):
            _character_frequency[_char] = \
                (_character_frequency.get(_char, 0) + _ct)

    _character_frequency = dict((zip(
        map(str, _character_frequency.keys()),
        map(int, _character_frequency.values())
    )))


    return _character_frequency


