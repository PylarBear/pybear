# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import numpy as np

from pybear.feature_extraction.text._TextLookup.TextLookupRealTime import \
    TextLookupRealTime



if __name__ == '__main__':

    _new_X = np.random.randint(0, 10, (15,))
    _new_X = np.array(list(map(str, _new_X)))
    _new_X = _new_X.reshape((5, 3))

    # skip_numbers = True
    # this proves that TextLookup does recognize str(numbers) as
    # numbers and 'skip_numbers' works

    _kwargs = {
        'update_lexicon': True,
        'skip_numbers': True,
        'auto_split': True,
        'auto_add_to_lexicon': False,
        'auto_delete': False,
        'DELETE_ALWAYS': None,
        'REPLACE_ALWAYS': None,
        'SKIP_ALWAYS': None,
        'SPLIT_ALWAYS': None,
        'verbose': False
    }

    TLRT = TextLookupRealTime(**_kwargs)
    TLRT.set_params(skip_numbers=True)
    out = TLRT.transform(_new_X)

    [print(_) for _ in out]







