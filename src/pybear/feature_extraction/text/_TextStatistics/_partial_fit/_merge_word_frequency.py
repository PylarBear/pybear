# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from .._type_aliases import WordFrequencyType

from .._validation._word_frequency import _val_word_frequency


def _merge_word_frequency(
    _current_word_frequency: WordFrequencyType,
    _word_frequency: WordFrequencyType
) ->  WordFrequencyType:

    """
    Merge the uniques and counts in the current partial fit's word
    frequency dictionary with the uniques and counts found in all
    previous fits of the TextStatistics instance.


    Parameters
    ----------
    _current_word_frequency:
        dict[str, int] - the unique words and their counts found in the
        current partial fit.
    _word_frequency:
        dict[str, int] - the unique words and their counts found in all
        previous fits on the TextStatistics instance.


    Return
    ------
    -
        _word_frequency: dict[str, int] - the merged uniques and counts
        for all words seen across of fits of the TextStatistics instance.


    """


    _val_word_frequency(_current_word_frequency)

    _val_word_frequency(_word_frequency)


    for k, v in _current_word_frequency.items():

        if k in _word_frequency:
            _word_frequency[k] += v
        elif k not in _word_frequency:
            _word_frequency[k] = v


    return _word_frequency







