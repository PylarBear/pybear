# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.feature_extraction.text import (
    AutoTextCleaner as ATC,
    TextReplacer,
    TextSplitter,
    TextStripper
)


import re



if __name__ == '__main__':


    _files = ['notepad.txt']

    TStrip = TextStripper()
    TRepl = TextReplacer(replace=(re.compile('[^a-zA-Z]'), ''))
    TSplit = TextSplitter(sep=' ')

    Trfm = ATC(
        global_sep=' ',
        case_sensitive=False,
        global_flags=None,
        remove_empty_rows=True,
        return_dim=1,
        strip=True,
        replace=(re.compile('[^a-zA-Z]'), ''),
        remove=(''),
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
        with open(rf'./{file}', 'r') as f:
            for line in f:
                _words.append(line)

        print(f'    splitting _words...')
        _words = TSplit.fit_transform(_words)

        # print(f'    replacing junk...')
        # _words = TRepl.fit_transform(_words)

        print(f'    stripping _words...')
        for r_idx, _line in enumerate(_words):
            try:
                float("".join(_line))
                _words[r_idx] = []
            except:
                _words[r_idx] = TStrip.fit_transform(_words[r_idx])

        _words = list(map(str, _words))

        print(f'    running ATC...')
        out = Trfm.transform(_words)


    # print(Trfm.lexicon_lookup_.LEXICON_ADDENDUM_)
    for line in out:
        print(line)


