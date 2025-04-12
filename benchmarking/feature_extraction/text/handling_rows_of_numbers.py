# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import operator
import re


from pybear.feature_extraction.text import (
    NGramMerger as NGM,
    StopRemover as SR,
    TextJoiner as TJO,
    TextJustifier as TJU,
    TextLookup as TL,
    TextNormalizer as TN,
    TextPadder as TP,
    TextRemover as TRem,
    TextReplacer as TRep,
    TextSplitter as TSpl,
    TextStatistics as TStat,
    TextStripper as TStrip
)


# 25_04_12 there was a problem with all of the modules failing on string
# data that looks like ['8\n', '9\n']. Only when there are no other
# legit strings. the problem appears to be in nan_mask_numerical.




if __name__ == '__main__':


    _TStrip = TStrip()
    _TSpl = TSpl(sep=' ')
    _TJO = TJO(sep=' ')
    _TN = TN(upper=True)
    _TL = TL(update_lexicon=True, auto_add_to_lexicon=True, skip_numbers=True, auto_split=True)
    _SR = SR(remove_empty_rows=True, n_jobs=-1)
    _NGM = NGM(ngrams=(('a', 'b'), ))
    _TJU = TJU(n_chars=79, sep=' ')
    _TRep = TRep(replace=(re.compile('[^a-zA-Z]'), ''))
    _TRem = TRem(remove='')
    _TStat = TStat(store_uniques=True)
    _TP = TP(fill='NA')


    _words = list(map(str, range(10)))
    words1 = list(map(operator.methodcaller('__add__', '\n'), _words))
    print(words1)
    words2 = [words1]

    names = \
        ['Stripper', 'Splitter', 'Joiner', 'Normalizer', 'Lookup', 'StopRemover',
         'NGrams', 'Justifier', 'Replacer', 'Remover', 'Statistics', 'Padder']

    modules = \
        [_TStrip, _TSpl, _TJO, _TN, _TL, _SR,
         _NGM, _TJU, _TRep, _TRem, _TStat, _TP]


    for _name, _module in zip(names, modules):

        for _dim in [1, 2]:

            # skip known dim blocks -- -- -- -- -- -- -- -- -- --
            if _name == 'Splitter' and _dim == 2:
                continue
            if _dim == 1 and _name in ['Joiner', 'Lookup', 'NGrams', 'Padder', 'StopRemover']:
                continue
            # END skip known dim blocks -- -- -- -- -- -- -- -- -- --

            if _dim == 1:
                _X = words1
            elif _dim == 2:
                _X = words2
            else:
                raise Exception

            try:
                _module.fit_transform(_X)
                print(f'\033[92m{_name} {_dim}D OK!\033[0m')
            except Exception as e:
                print(f'\033[91m{_name} {_dim}D FAIL' + f'---' + str(e) +f'\033[0m')






