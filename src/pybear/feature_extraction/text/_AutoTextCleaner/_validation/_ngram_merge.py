# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from .._type_aliases import NGramsType

# this is directly from NGramMerger
from ..._NGramMerger._validation._ngrams import _val_ngrams



def _val_ngram_merge(
    _ngram_merge: NGramsType
) -> None:

    """
    Validate ngram_merge. The series of string literals and/or re.compile
    objects that specify an n-gram. Can be None.


    Parameters
    ----------
    _ngram_merge:
        Union[Sequence[Sequence[Union[str, re.Pattern[str]]]], None] - A
        sequence of sequences, where each inner sequence holds a series
        of string literals and/or re.compile objects that specify an
        n-gram. Cannot be empty, and cannot have any n-grams with less
        than 2 entries. Can be None.


    Returns
    -------
    -
        None

    """


    _val_ngrams(_ngram_merge)





