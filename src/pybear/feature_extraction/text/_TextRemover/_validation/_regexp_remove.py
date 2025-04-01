# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing_extensions import Union

import re
import numbers



def _val_regexp_remove(
    _rr: Union[None, re.Pattern[str], tuple[re.Pattern[str], ...],
            list[Union[None, re.Pattern[str], tuple[re.Pattern[str], ...]]]],
    _n_rows: numbers.Integral
) -> None:

    """
    Validate the WIP parameter 'regexp_remove'. Must be None, re.Pattern,
    a tuple of re.Patterns, or a list of Nones, re.Patterns, and/or
    tuples of re.Patterns.


    Parameters
    ----------
    _rr:
        Union[None, re.Pattern, tuple[re.Pattern, ...],
        list[Union[None, re.Pattern, tuple[re.Pattern, ...]]]] - the
        regex pattern(s) to remove for a 1D list of strings.
    _n_rows:
        numbers.Integral - the number of rows in the data passed to
        transform.


    Returns
    -------
    -
        None

    """


    assert isinstance(_n_rows, numbers.Integral)
    assert not isinstance(_n_rows, bool)
    assert _n_rows >= 0


    try:
        if isinstance(_rr, (type(None), re.Pattern)):
            raise UnicodeError
        elif isinstance(_rr, tuple):
            if not all(map(isinstance, _rr, (re.Pattern for _ in _rr))):
                raise Exception
            raise UnicodeError
        elif isinstance(_rr, list):
            for thing in _rr:
                if isinstance(thing, (type(None), re.Pattern)):
                    pass
                elif isinstance(thing, tuple):
                    if not all(map(isinstance, thing, (re.Pattern for _ in thing))):
                        raise Exception
                else:
                    raise Exception

            if len(_rr) != _n_rows:
                raise TimeoutError

            raise UnicodeError
        else:
            raise Exception
    except UnicodeError:
        pass
    except TimeoutError:
        raise ValueError(
            f"if 'regexp' is a list, the length must be equal to the number "
            f"of rows in the data passed to transform. \ngot {len(_rr)}, "
            f"expected {_n_rows}"
        )
    except Exception as e:
        raise TypeError(
            f"'regexp_remove' must None, re.Pattern, tuple[re.Pattern, ...], "
            f"or a list of Nones, re.Patterns, and/or tuple[re.Pattern, ...]."
            f"\ngot {type(_rr)}."
        )






