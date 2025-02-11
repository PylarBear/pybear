# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from .._type_aliases import StringFrequencyType

from .._validation._string_frequency import _val_string_frequency


def _merge_string_frequency(
    _current_string_frequency: StringFrequencyType,
    _string_frequency: StringFrequencyType
) ->  StringFrequencyType:

    """
    Merge the uniques and counts in the current partial fit's string
    frequency dictionary with the uniques and counts found in all
    previous partial fits of the TextStatistics instance.


    Parameters
    ----------
    _current_string_frequency:
        dict[str, numbers.Integral] - the unique strings and their counts
        found in the current partial fit.
    _string_frequency:
        dict[str, numbers.Integral] - the unique strings and their counts
        found in all previous partial fits on the TextStatistics instance.


    Return
    ------
    -
        _string_frequency: dict[str, numbers.Integral] - the merged
        uniques and counts for all strings seen across all partial fits
        of the TextStatistics instance.


    """


    _val_string_frequency(_current_string_frequency)

    # _string_frequency will be {} on first pass
    _val_string_frequency(_string_frequency)


    # pizza maybe do some benchmarking on this
    for k, v in _current_string_frequency.items():

        _string_frequency[str(k)] = (_string_frequency.get(str(k), 0) + v)

        # was:
        # if k in _string_frequency:
        #     _string_frequency[k] += v
        # else:
        #     _string_frequency[k] = v


    return _string_frequency







