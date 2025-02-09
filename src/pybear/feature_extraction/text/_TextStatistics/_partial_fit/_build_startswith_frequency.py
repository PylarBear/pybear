# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import numbers

from .._validation._word_frequency import _val_word_frequency



def _build_startswith_frequency(
    _word_frequency: dict[str, numbers.Integral]
) -> dict[str, numbers.Integral]:

    """
    Build a dictionary that contains the first character of every string
    in '_word_frequency' as keys and the number of times that that
    character appears as the first character of a string as the values.


    Parameters
    ----------
    _word_frequency:
        dict[str, numbers.Integral] - the dictionary holding the unique
        strings passed to the current partial fit and their respective
        frequencies.


    Return
    ------
    -
        _startswith_frequency: dict[str, numbers.Integral] - a dictionary
        that holds the first characters of every string passed to
        this partial fit and their respective number of appearances in
        the first position as values.


    """


    _val_word_frequency(_word_frequency)

    # pizza, maybe do some benchmarking on this.
    # another way would to pass WORDS directly and do:
    # _char_getter = map(lambda x: str(x[0]), WORDS)
    # _startswith_frequency: dict[str: numbers.Integral] = dict((zip(
    #     *np.unique(np.fromiter(_char_getter, dtype='<U1'), return_counts=True)
    # )))


    _startswith_frequency: dict[str: numbers.Integral] = {}

    for _string, _ct in _word_frequency.items():
        _startswith_frequency[str(_string[0])] = \
            (_startswith_frequency.get(str(_string[0]), 0) + _ct)

    _startswith_frequency = dict((zip(
        map(str, _startswith_frequency.keys()),
        map(int, _startswith_frequency.values())
    )))


    return _startswith_frequency


