# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Sequence
from typing_extensions import Union

import re



def _val_ngrams(
    _ngrams: Sequence[Sequence[Union[str, re.Pattern]]]
) -> None:

    """
    Validate ngrams. The series of string literals and/or re.compile
    objects that specify an n-gram.


    Parameters
    ----------
    _ngrams:
        Sequence[Sequence[Union[str, re.Pattern]]] - A sequence of
        sequences, where each inner sequence holds a series of string
        literals and/or re.compile objects that specify an n-gram.
        Cannot be empty, and cannot have any n-grams with less than 2
        entries.


    Returns
    -------
    -
        None

    """


    err_msg = (f"'ngrams' must be a 1D sequence of sequences of string "
               f"literals and/or re.compile objects. \ncannot be empty, "
               f"and cannot contain any n-gram sequences with less than "
               f"2 entries.")

    # this validates that the outer container is 1D iterable
    try:
        iter(_ngrams)
        if isinstance(_ngrams, (str, dict)):
            raise Exception
        if len(_ngrams) == 0:
            raise UnicodeError
    except UnicodeError:
        raise ValueError(err_msg)
    except Exception as e:
        raise TypeError(err_msg)

    # this validates the contents of the outer iterable
    for _inner in _ngrams:

        try:
            iter(_inner)
            if isinstance(_inner, (str, dict)):
                raise Exception
            if len(_inner) < 2:
                raise UnicodeError
            if not all(map(
                isinstance,
                _inner,
                ((str, re.Pattern) for _ in _inner)
            )):
                raise Exception
        except UnicodeError:
            raise ValueError(err_msg)
        except Exception as e:
            raise TypeError(err_msg)








