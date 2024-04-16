import numpy as np
import sys, inspect
from general_text import alphanumeric_str as ans, statistics as stats
from debug import get_module_name as gmn
from data_validation import arg_kwarg_validater as akv
from general_text.old_py_lexicon import lexicon_y as ly, lexicon_t as lt, lexicon_sn_sz as lsn, lexicon_z as lz, \
    lexicon_u as lu, lexicon_x as lx, lexicon_w as lw, lexicon_v as lv, lexicon_sa_sm as lsa, lexicon_n as ln, \
    lexicon_e as le, lexicon_r as lr, lexicon_d as ld, lexicon_f as lf, lexicon_m as lm, lexicon_a as la, \
    lexicon_p as lp, lexicon_l as ll, lexicon_c as lc, lexicon_j as lj, lexicon_h as lh, lexicon_b as lb, \
    lexicon_i as li, lexicon_o as lo, lexicon_g as lg, lexicon_k as lk, lexicon_q as lq



# _exception
# size
# find_duplicates
# check_order
# string_validation
# lookup_substring
# lookup_word
# statistics
# lexicon
# old_py_lexicon


class Lexicon:

    def __init__(self):
        self.this_module = gmn.get_module_name(str(sys.modules[__name__]))
        fxn = '__init__'

        self.LEXICON = self.lexicon()


    def _exception(self, fxn, words):
        fxn = f".{fxn}()" if not fxn is None else ""
        raise Exception(f'\n*** {self.this_module}{fxn} >>> {words} ***\n')


    def size(self):
        return len(self.LEXICON)


    def find_duplicates(self):
        # TESTED AND VERIFIED 1/19/23
        UNIQUES, COUNTS = np.unique(self.LEXICON, return_counts=True)
        if len(UNIQUES) == len(self.LEXICON):
            print(f'\n*** THERE ARE NO DUPLICATES IN THE LEXICON ***\n'); del UNIQUES, COUNTS
        else:
            MASTER_SORT = np.flip(np.argsort(COUNTS))
            MASK = MASTER_SORT[..., COUNTS[..., MASTER_SORT] > 1]; del MASTER_SORT
            MASKED_SORTED_UNIQUES = UNIQUES[..., MASK]; del UNIQUES
            MASKED_SORTED_COUNTS = COUNTS[..., MASK]; del COUNTS, MASK
            INDICES = np.unique(MASKED_SORTED_UNIQUES, return_index=True)[1]
            DUPLICATES = MASKED_SORTED_UNIQUES[INDICES]; del MASKED_SORTED_UNIQUES
            COUNTS = MASKED_SORTED_COUNTS[INDICES]; del MASKED_SORTED_COUNTS, INDICES

            if len(DUPLICATES) == 0:
                print(f'\n*** THERE ARE NO DUPLICATED IN LEXICON ***\n')
            else:
                print()
                print(f'*' * 100)
                print(f'\n DUPLICATE'.ljust(30) + f'COUNT')
                print(f'-' * 35)
                [print(f'{d}'.ljust(30) + f'{c}') for d,c in zip(DUPLICATES, COUNTS)]
                print()
                print(f'*' * 100)

            del DUPLICATES, COUNTS


    def check_order(self):
        _ = self.LEXICON
        __ = np.unique(self.LEXICON)

        if np.array_equiv(_, __):
            print(f'\n*** LEXICON IS IN ALPHABETICAL ORDER ***\n')
        else:
            OUT_OF_ORDER = []
            for idx in range(len(__)):
                if _[idx] != __[idx]:
                    OUT_OF_ORDER.append(__[idx])
            if len(OUT_OF_ORDER) > 0:
                print(f'OUT OF ORDER:')
                print(OUT_OF_ORDER)
            del OUT_OF_ORDER


    def string_validation(self, char_seq, fxn=None):
        fxn = fxn if not fxn is None else inspect.stack()[0][3]
        if char_seq is None: self._exception(fxn, f'MUST ENTER A CHARACTER SEQUENCE TO SELECT WORDS BY')
        if not isinstance(char_seq, str): self._exception(fxn, f'char_seq MUST BE A str OF ALPHAS')
        # CRAZY WAY OF VALIDATION
        [_ if _.upper() in ans.alphabet_str_upper() else
                        self._exception(fxn, f'char_seq MUST ONLY CONTAIN ALPHA CHARACTERS') for _ in char_seq]


    def lookup_substring(self, char_seq=None):

        fxn = inspect.stack()[0][3]

        self.string_validation(char_seq, fxn=fxn)

        # IF GET THIS FAR:
        char_seq = char_seq.upper()
        MASK = np.fromiter(map(lambda x: x.find(char_seq, 0, len(char_seq)) + 1, self.LEXICON), dtype=bool)
        SELECTED_WORDS = self.lexicon()[MASK]
        del char_seq, MASK

        return SELECTED_WORDS


    def lookup_word(self, char_seq=None, bypass_validation=False):

        fxn = inspect.stack()[0][3]

        bypass_validation = akv.arg_kwarg_validater(bypass_validation, 'bypass_validation', [True, False, None],
                                                    self.this_module, fxn, return_if_none=True)

        if not bypass_validation:
            self.string_validation(char_seq, fxn=fxn)

        return char_seq.upper() in self.LEXICON


    def statistics(self):
        stats.statistics(self.LEXICON)


    def lexicon(self):
        base_path = r'C:/Users/Bill/AppData/Local/Programs/Python/Python39/PyCharmProjects/general_text/lexicon/'
        # base_path = r'lexicon/'
        file_base = f'lexicon_'



        _open = lambda letter: np.char.split(
                                    np.char.replace(
                                        open(base_path + file_base + letter.lower() + '.txt', encoding="utf8").read(),
                                        f'\n',
                                        f' '
                                    )
                                ).reshape((1,-1))[0][0]

        return np.hstack((
                            _open('a'),
                            _open('b'),
                            _open('c'),
                            _open('d'),
                            _open('e'),
                            _open('f'),
                            _open('g'),
                            _open('h'),
                            _open('i'),
                            _open('j'),
                            _open('k'),
                            _open('l'),
                            _open('m'),
                            _open('n'),
                            _open('o'),
                            _open('p'),
                            _open('q'),
                            _open('r'),
                            _open('s'),
                            _open('t'),
                            _open('u'),
                            _open('v'),
                            _open('w'),
                            _open('x'),
                            _open('y'),
                            _open('z')
                        ))


    def old_py_lexicon(self):
            return np.hstack((
                            la.lexicon_a(),
                            lb.lexicon_b(),
                            lc.lexicon_c(),
                            ld.lexicon_d(),
                            le.lexicon_e(),
                            lf.lexicon_f(),
                            lg.lexicon_g(),
                            lh.lexicon_h(),
                            li.lexicon_i(),
                            lj.lexicon_j(),
                            lk.lexicon_k(),
                            ll.lexicon_l(),
                            lm.lexicon_m(),
                            ln.lexicon_n(),
                            lo.lexicon_o(),
                            lp.lexicon_p(),
                            lq.lexicon_q(),
                            lr.lexicon_r(),
                            lsa.lexicon_sa_sm(),
                            lsn.lexicon_sn_sz(),
                            lt.lexicon_t(),
                            lu.lexicon_u(),
                            lv.lexicon_v(),
                            lw.lexicon_w(),
                            lx.lexicon_x(),
                            ly.lexicon_y(),
                            lz.lexicon_z()
                ))



if __name__ == '__main__':
    Lexicon().check_order()
    Lexicon().find_duplicates()
    print(Lexicon().size())
    Lexicon().statistics()

