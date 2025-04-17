# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.feature_extraction.text import AutoTextCleaner as ATC

import os
import re



if __name__ == '__main__':


    _files = ['notepad.txt']

    Trfm = ATC(
        global_sep=' ',
        case_sensitive=False,
        global_flags=None,
        remove_empty_rows=False,
        return_dim=1,
        strip=True,
        replace=(re.compile('[^0-9a-zA-Z]'), ''),
        remove=None,
        normalize=True,
        lexicon_lookup='manual',
        remove_stops=False,
        ngram_merge=None,
        justify=None,
        get_statistics=None
    )


    for file in _files:

        print(f'running {file}...')
        print(f'    building _words...')
        _words = []
        with open(os.path.join(os.curdir, rf'{file}'), 'r') as f:
            for line in f:
                _words.append(line)

        print(f'    running ATC...')
        out = Trfm.transform(_words)


    # print(Trfm.lexicon_lookup_.LEXICON_ADDENDUM_)
    for line in out:
        print(line)






