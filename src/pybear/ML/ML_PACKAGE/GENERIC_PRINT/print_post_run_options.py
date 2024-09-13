import numpy as n


# BEAR THESE ALL DISAPPEAR AFTER FULL ROLLOUT OF dict_menu_print()

class SelectionsPrint():
    def __init__(self, COMMANDS, options, append_ct_limit=4, max_len=''):
        self.COMMANDS = COMMANDS
        self.post_run_options = options
        self.append_ct_limit = append_ct_limit
        self.max_len = max_len
        print()

        display_str = ''
        _width = max(map(len, self.COMMANDS)) if self.max_len == '' else self.max_len
        for itr, command in enumerate(self.COMMANDS):
            if True in map(lambda x: f'({x})' in command.upper(), options.upper()):
                display_str += f'{command}'.ljust(_width + 2)
            else: itr -= 1
            if itr % self.append_ct_limit == self.append_ct_limit-1:
                print(display_str)
                display_str = ''
        else:
            if not display_str == '': print(display_str)

        del _width, display_str


# CALLED BY NN
class TopLevelMenuPrint(SelectionsPrint):
    pass


class SelectionsPrintOnly:
    def __init__(self, COMMANDS, append_ct_limit=4, max_len=''):
        self.COMMANDS = COMMANDS
        self.append_ct_limit = append_ct_limit
        self.max_len = max_len

        print()

        display_str, append_ctr = '', 0
        _width = max(map(len, self.COMMANDS)) if self.max_len == '' else self.max_len
        for itr, command in enumerate(self.COMMANDS):
            display_str += f'{command}'.ljust(_width + 2)
            if itr % self.append_ct_limit == self.append_ct_limit-1:
                print(display_str)
                display_str = ''
        else:
            if not display_str=='': print(display_str)
        del _width, display_str






if __name__ == '__main__':

    TEST_LIST = ['accept config / continue / bypass(a)',
             'abandon session changes and exit(b)',
             'print frequencies(c)',
             'print a preview of objects as DFs(p)',
             'print column names & data type(q)',
             'start over - session reset(r)',
             'call a standard config(s)',
             'start over - global reset(t)',
             'placeholder(y)',
             'placeholder(z)',
             'get object types(1)',
             'stinks'
     ]

    list_str = 'ABCPQRSTYZ1'

    print(f'\nSTART TEST PRINT OF LIST...')
    SelectionsPrint(TEST_LIST, list_str, append_ct_limit=3, max_len=40)
    print(f'\nTEST PRINT OF LIST COMPLETE.')


    TEST_DICT = {'a': 'accept config / continue / bypass',
                 'b': 'abandon session changes and exit',
                 'c': 'print frequencies',
                 'p': 'print a preview of objects as DFs',
                 'q': 'print column names & data type',
                 'r': 'start over - session reset',
                 's': 'call a standard config',
                 't': 'start over - global reset',
                 'y': 'placeholder',
                 'z': 'placeholder',
                 '1': 'get object types'
     }

    print(f'\nSTART TEST PRINT OF DICT...')
    SelectionsPrint(list({f'{v}({k})' for k,v in TEST_DICT.items()}),
                    "".join(TEST_DICT.keys()).upper(), append_ct_limit=3, max_len=40)
    print(f'\nTEST PRINT OF DICT COMPLETE.')











