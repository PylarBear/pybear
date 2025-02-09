# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from .._type_aliases import OverallStatisticsType

import numbers



def _val_overall_statistics_dict(
    _overall_statistics_dict: OverallStatisticsType
) -> None:

    """
    Validate overall_statistics_dict is a dictionary with the required
    keys and valid values.


    Parameters
    ----------
    _overall_statistics_dict:
        dict[str, numbers.Real] - the dictionary containing summary
        statistics about the words fit on the TextStatistics instance,
        such as number of words, average length of words, etc.


    Return
    ------
    -
        None


    """

    assert isinstance(_overall_statistics_dict, dict)
    assert len(_overall_statistics_dict) == 6

    _allowed_keys = [
        'size',
        'uniques_count',
        'max_length',
        'min_length',
        'average_length',
        'std_length'
    ]

    for key in _overall_statistics_dict:
        if key not in _allowed_keys:
            raise AssertionError(
                f"dict key '{key}' not an allowed key for overall_statistics_dict"
            )

    _size = _overall_statistics_dict['size']
    _uniques_count =_overall_statistics_dict['uniques_count']
    _max_len = _overall_statistics_dict['max_length']
    _min_len = _overall_statistics_dict['min_length']
    _average_length = _overall_statistics_dict['average_length']
    _std_length = _overall_statistics_dict['std_length']

    assert isinstance(_size, int)
    assert not isinstance(_size, bool)
    assert _size >= 0

    assert isinstance(_uniques_count, int)
    assert not isinstance(_uniques_count, bool)
    assert _uniques_count >= 0
    assert _uniques_count <= _size

    assert isinstance(_max_len, int)
    assert not isinstance(_max_len, bool)
    assert _max_len >= 0

    assert isinstance(_min_len, int)
    assert not isinstance(_min_len, bool)
    assert _min_len >= 0
    assert _min_len <= _max_len

    assert isinstance(_average_length, numbers.Real)
    assert not isinstance(_average_length, bool)
    assert _average_length <= _max_len
    assert _average_length >= _min_len

    assert isinstance(_std_length, numbers.Real)
    assert not isinstance(_std_length, bool)
    assert _std_length >= 0




