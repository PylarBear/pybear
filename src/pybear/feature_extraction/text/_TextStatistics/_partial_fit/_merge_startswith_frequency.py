# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from .._type_aliases import StartsWithFrequencyType

from .._validation._startswith_frequency import _val_startswith_frequency



def _merge_startswith_frequency(
    _current_startswith_frequency: StartsWithFrequencyType,
    _startswith_frequency: StartsWithFrequencyType
) -> StartsWithFrequencyType:

    """
    Merge the unique first characters and counts in the current partial
    fit's startswith frequency dictionary with those found in all
    previous partial fits of the TextStatistics instance.


    Parameters
    ----------
    _current_startswith_frequency:
        dict[str, numbers.Integral] - the unique first characters and
        their counts found in the current partial fit.
    _startswith_frequency:
        dict[str, numbers.Integral] - the unique first characters and
        their counts found in all previous partial fits on the
        TextStatistics instance.


    Return
    ------
    -
        _startswith_frequency: dict[str, numbers.Integral] - the merged
        unique first characters and counts for all words seen across all
        partial fits of the TextStatistics instance.


    """

    _val_startswith_frequency(_current_startswith_frequency)

    # _startswith_frequency will be {} on first pass
    _val_startswith_frequency(_startswith_frequency)


    # pizza, maybe do some benchmarking on this.
    # another way would to pass WORDS directly and do:
    # _char_getter = map(lambda x: str(x[0]), WORDS)
    # _startswith_frequency: dict[str: numbers.Integral] = dict((zip(
    #     *np.unique(np.fromiter(_char_getter, dtype='<U1'), return_counts=True)
    # )))

    # pizza maybe do some benchmarking on this
    for k, v in _current_startswith_frequency.items():

        _startswith_frequency[str(k)] = (_startswith_frequency.get(str(k), 0) + v)

        # was:
        # if k in _startswith_frequency:
        #     _startswith_frequency[k] += v
        # else:
        #     _startswith_frequency[k] = v


    return _startswith_frequency





