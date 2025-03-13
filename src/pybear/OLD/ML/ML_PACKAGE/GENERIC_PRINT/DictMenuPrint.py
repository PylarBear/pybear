import sys, inspect
from debug import get_module_name as gmn
from data_validation import validate_user_input as vui




class DictMenuPrint:
    """Validates inputs and displays menu."""
    def __init__(self, MENU_DICT, disp_len=140, fixed_col_width=None, allowed=None, disallowed=None):

        self.this_module = gmn.get_module_name(str(sys.modules[__name__]))
        fxn = '__init__'

        if not isinstance(disp_len, int): self._exception(f'disp_len MUST BE AN INTEGER')

        if not isinstance(fixed_col_width, int) and not fixed_col_width is None:
            self._exception(f'fixed_col_width MUST BE AN INTEGER OR None')

        if max(map(len, MENU_DICT.keys())) != 1: self._exception(f'ILLEGAL KEY IN MENU DICT', fxn)

        all_allowed_str = ''.join(MENU_DICT.keys()).upper()

        allowed = None if allowed == '' else allowed
        disallowed = None if disallowed == '' else disallowed

        if not allowed is None and not disallowed is None:
            self._exception(f'CANNOT ENTER BOTH allowed AND disallowed, MUST BE ONE OR THE OTHER OR NEITHER', fxn)

        elif not allowed is None:
            if not isinstance(allowed, str): self._exception(f'allowed KWARG REQUIRES str AS INPUT', fxn)

            allowed = allowed.upper()

            # CLEAR OUT ANY DUPLICATES THAT MAY BE IN ALLOWED
            allowed = ''.join([_ for _ in all_allowed_str if _ in allowed]).upper()

            if False in map(lambda x: x in all_allowed_str, allowed):
                self._exception(f'INVALID KEY IN allowed ({allowed}), MUST BE IN {all_allowed_str}.', fxn)

        elif not disallowed is None:
            if not isinstance(disallowed, str): self._exception(f'disallowed KWARG REQUIRES str AS INPUT', fxn)

            disallowed = disallowed.upper()

            if False in map(lambda x: x in all_allowed_str, disallowed):
                self._exception(f'INVALID KEY IN disallowed ({disallowed}), MUST BE IN {all_allowed_str}.', fxn)

            # SERENDIPITOUSLY HANDLES ANY DUPLICATES THAT MAY BE IN DISALLOWED
            allowed = ''.join([_ for _ in all_allowed_str if not _ in disallowed]).upper()
            del disallowed

        elif allowed is None and disallowed is None:
            allowed = all_allowed_str

        WIP_DICT = {k: v for k, v in MENU_DICT.items() if k.upper() in allowed}

        if not fixed_col_width is None:
            _ljust = int(fixed_col_width)
            num_cols = disp_len // _ljust
            disp_len = fixed_col_width
        else:  # fixed_col_width is None
            num_cols = max(1, disp_len // (max(map(len, WIP_DICT.values())) + 3 + 2))   # +3 FOR (_) +2 FOR BUFFER
            _ljust = disp_len // num_cols

        print_line = ''
        for itr, (k, v) in enumerate(WIP_DICT.items()):
            print_line += f'{v[:disp_len-5]}({k.lower()})'.ljust(_ljust)
            if itr % num_cols == num_cols - 1 or itr == len(WIP_DICT)-1: print(print_line); print_line = ''

        del num_cols, print_line, WIP_DICT, all_allowed_str

        self.allowed = allowed


    def _exception(self, text, fxn=None):
        fxn = f'.{fxn}()' if not fxn is None else ''
        raise Exception(f'\033[91m*** {self.this_module}{fxn} >>> {text} ***\033[0m')


    def select(self, prompt):
        """Returns single-letter menu command."""
        return vui.validate_user_str(f'\n{prompt} > ', self.allowed)













if __name__ == '__main__':

    # DictMenuPrint(DICT, disp_len=120, allowed=None, disallowed=None)

    test_start = lambda name: print(f'\nSTARTING {name}...')
    test_pass = lambda name: print(f'\033[92m*** {name} PASSED ***\033[0m')
    test_fail = lambda name, reason: print(f'\033[91m*** {name} FAILED. REASON = {reason} ***\033[0m')

    # TEST disp_len VALIDATION ##########################################################################################
    name = 'disp_len ACCEPTS VALID INT'
    test_start(name)
    try:
        DICT = {'A':'Test 1', 'B': 'Test 2'}
        DictMenuPrint(DICT, disp_len=140)
        test_pass(name)
    except:
        test_fail(name, 'REJECTED A VALID INTEGER')

    name = 'disp_len REJECTS NON-INTEGER'
    test_start(name)
    try:
        DICT = {'A':'Test 1', 'B': 'Test 2'}
        DictMenuPrint(DICT, disp_len='TEST')
        test_fail(name, 'ACCEPTED AN INVALID NON-INTEGER')
    except:
        test_pass(name)
    # END TEST disp_len VALIDATION ######################################################################################

    # TEST fixed_col_width VALIDATION ##########################################################################################
    name = 'fixed_col_width ACCEPTS VALID INT'
    test_start(name)
    try:
        DICT = {'A':'Test 1', 'B': 'Test 2'}
        DictMenuPrint(DICT, fixed_col_width=40)
        test_pass(name)
    except:
        test_fail(name, 'REJECTED A VALID INTEGER')

    name = 'fixed_col_width REJECTS NON-INTEGER'
    test_start(name)
    try:
        DICT = {'A':'Test 1', 'B': 'Test 2'}
        DictMenuPrint(DICT, fixed_col_width='TEST')
        test_fail(name, 'ACCEPTED AN INVALID NON-INTEGER')
    except:
        test_pass(name)
    # END TEST fixed_col_width VALIDATION ######################################################################################

    # TEST EXCEPTS FOR BAD KEY ##########################################################################################
    name = 'DICT KEY len != 1'
    test_start(name)
    try:
        DICT = {'AA':'Test 1', 'B': 'Test 2'}
        DictMenuPrint(DICT)
        test_fail(name, 'PASSED A 2-CHARACTER KEY')
    except:
        test_pass(name)
    # END TEST EXCEPTS FOR BAD KEY ######################################################################################

    # TEST EXCEPTS NON-str allowed #######################################################################################
    name = 'NON-str allowed'
    test_start(name)
    try:
        DICT = {'A':'Test 1', 'B': 'Test 2', 'C': 'Test 3'}
        DictMenuPrint(DICT, allowed=['A'])
        test_fail(name, 'PASSED A NON-str FOR allowed')
    except:
        test_pass(name)
    # END TEST EXCEPTS BAD allowed ######################################################################################

    # TEST PASSES GOOD allowed ##########################################################################################
    name = 'GOOD allowed'
    test_start(name)
    try:
        DICT = {'A':'Test 1', 'B': 'Test 2', 'C': 'Test 3'}
        DictMenuPrint(DICT, allowed='A')
        test_pass(name)
    except:
        test_fail(name, sys.exc_info()[1])
    # END TEST PASSES GOOD allowed ######################################################################################

    # TEST EXCEPTS BAD allowed ##########################################################################################
    name = 'BAD allowed'
    test_start(name)
    try:
        DICT = {'A':'Test 1', 'B': 'Test 2', 'C': 'Test 3'}
        DictMenuPrint(DICT, allowed='D')
        test_fail(name, 'PASSED A BAD allowed CHAR')
    except:
        test_pass(name)
    # END TEST EXCEPTS BAD allowed ######################################################################################

    # TEST EXCEPTS NON-str disallowed #######################################################################################
    name = 'NON-str disallowed'
    test_start(name)
    try:
        DICT = {'A':'Test 1', 'B': 'Test 2', 'C': 'Test 3'}
        DictMenuPrint(DICT, disallowed=['A'])
        test_fail(name, 'PASSED A NON-str FOR disallowed')
    except:
        test_pass(name)
    # END TEST EXCEPTS BAD allowed ######################################################################################

    # TEST PASSES GOOD disallowed ##########################################################################################
    name = 'GOOD disallowed'
    test_start(name)
    try:
        DICT = {'A':'Test 1', 'B': 'Test 2', 'C': 'Test 3'}
        DictMenuPrint(DICT, disallowed='A')
        test_pass(name)
    except:
        test_fail(name, sys.exc_info()[1])
    # END TEST PASSES GOOD allowed ######################################################################################

    # TEST EXCEPTS BAD disallowed ##########################################################################################
    name = 'BAD disallowed'
    test_start(name)
    try:
        DICT = {'A':'Test 1', 'B': 'Test 2', 'C': 'Test 3'}
        DictMenuPrint(DICT, disallowed='D')
        test_fail(name, 'PASSED A BAD disallowed CHAR')
    except:
        test_pass(name)
    # END TEST EXCEPTS BAD allowed ######################################################################################

    # TEST EXCEPTS allowed & disallowed PASSED #########################################################################
    name = 'allowed & disallowed GIVEN'
    test_start(name)
    try:
        DICT = {'A':'Test 1', 'B': 'Test 2', 'C': 'Test 3'}
        DictMenuPrint(DICT, allowed='A', disallowed='B')
        test_fail(name, 'PASSED WHEN BOTH allowed & disallowed GIVEN')
    except:
        test_pass(name)
    # END TEST EXCEPTS allowed & disallowed PASSED #########################################################################

    # TEST DISPLAY OF MANY SMALL OPTIONS ##############################################################################
    name = 'DISPLAY OF MANY SMALL OPTIONS'
    test_start(name)

    VALUES = [f'Test{_}' for _ in range(0,27)]
    DICT = dict((zip([*'ABCDEFGHIJKLMNOPQURSTUVWXYZ'], VALUES)))
    DictMenuPrint(DICT, disp_len=50)
    # END TEST DISPLAY OF MANY SMALL OPTIONS ##############################################################################

    # TEST DISPLAY OF MANY SMALL OPTIONS ##############################################################################
    name = 'DISPLAY OF MANY SMALL OPTIONS'
    test_start(name)

    VALUES = [f'Test{_}' for _ in range(0,27)]
    DICT = dict((zip([*'ABCDEFGHIJKLMNOPQURSTUVWXYZ'], VALUES)))
    DictMenuPrint(DICT, disp_len=140)
    # END TEST DISPLAY OF MANY SMALL OPTIONS ##############################################################################

    # TEST DISPLAY OF MANY MEDIUM OPTIONS ##############################################################################
    name = 'DISPLAY OF MANY MEDIUM OPTIONS'
    test_start(name)

    VALUES = [f'Test of many medium-sized options{_}' for _ in range(0,27)]
    DICT = dict((zip([*'ABCDEFGHIJKLMNOPQURSTUVWXYZ'], VALUES)))
    DictMenuPrint(DICT, disp_len=140)
    # END TEST DISPLAY OF MANY MEDIUM OPTIONS ###########################################################################

    # TEST DISPLAY OF LONG OPTIONS ######################################################################################
    name = 'DISPLAY OF EXTREMELY LONG OPTIONS'
    test_start(name)

    VALUES = [f'Test of extremely long options, so long they go past the disp_len, so they should be truncated{_}' for _ in range(0,27)]
    DICT = dict((zip([*'ABCDEFGHIJKLMNOPQURSTUVWXYZ'], VALUES)))
    DictMenuPrint(DICT, disp_len=90)
    # END TEST DISPLAY OF LONG OPTIONS ##################################################################################


    # TEST fixed_col_width DISPLAYS CORRECTLY ###################################################################################
    name = 'TEST fixed_col_width DISPLAYS CORRECTLY'
    test_start(name)
    SHORT_DICT = {'a':'Test 1', 'b':'Test 2', 'c':'Test 3'}
    MEDIUM_DICT = {'a':'Medium len text Test 1', 'b':'Medium len text Test 2', 'c':'Medium len text Test 3'}
    LONG_DICT = {'a': 'Print long display of text Test 1', 'b': 'Print long display of text text Test 2', 'c': 'Print long display of text Test 3'}
    for name, SAMPLE_DICT in zip(('SHORT_DICT', 'MEDIUM_DICT', 'LONG_DICT'),(SHORT_DICT,MEDIUM_DICT,LONG_DICT)):
        print(f'{name}:')
        DictMenuPrint(SAMPLE_DICT, fixed_col_width=40)
        print()

    print(f'ALL PRINTOUTS SHOULD HAVE SAME MARGINS')
    # END TEST fixed_col_width DISPLAYS CORRECTLY ###############################################################################


    # TEST allowed DISPLAYS CORRECTLY ###################################################################################
    name = 'TEST allowed DISPLAYS CORRECTLY'
    test_start(name)
    DICT = {'a':'Test 1', 'b':'Test 2', 'c':'Test 3'}
    DictMenuPrint(DICT, allowed='ac')

    __ = input(f'Only a AND c SHOULD BE SHOWN, HIT ENTER TO CONTINUE > ')
    # END TEST allowed DISPLAYS CORRECTLY ###############################################################################


    # TEST disallowed DISPLAYS CORRECTLY ###################################################################################
    name = 'TEST disallowed DISPLAYS CORRECTLY'
    test_start(name)
    DICT = {'a':'Test 1', 'b':'Test 2', 'c':'Test 3'}
    DictMenuPrint(DICT, disallowed='c')

    __ = input(f'Only a AND b SHOULD BE SHOWN, HIT ENTER TO CONTINUE > ')
    # END TEST allowed DISPLAYS CORRECTLY ###############################################################################


    # TEST select RETURNS CORRECTLY ######################################################################################
    name = 'TEST select RETURNS CORRECTLY'
    test_start(name)
    DICT = {'a':'Test 1', 'b':'Test 2', 'c':'Test 3'}
    TestClass = DictMenuPrint(DICT)
    _ = TestClass.select(f'Enter selection')
    print(f'TestClass.select RETURNED "{_}"')
    # END TEST select RETURNS CORRECTLY ######################################################################################













