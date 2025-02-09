# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from .._type_aliases import WordFrequencyType



def _val_word_frequency(
    _word_frequency: WordFrequencyType
) -> None:

    """
    Validate the word_frequency dictionary
    - is a dictionary
    - has strings for keys
    - has non-bool integers for values, and all values are >= 1


    Parameters
    ----------
    _word_frequency:
        dict[str, numbers.Integral] - a dictionary of unique character
        strings and counts.


    Return
    ------
    -
        None


    """


    assert isinstance(_word_frequency, dict)
    for k, v in _word_frequency.items():
        assert isinstance(k, str)
        assert isinstance(v, int)
        assert not isinstance(v, bool)
        assert v >= 1








