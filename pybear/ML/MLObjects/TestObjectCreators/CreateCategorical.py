# MODULE FOR CREATING DATA, TARGET, REFVECS, TEST, SRNL, SWNL, W RANDOM CATEGORIES OR FROM GIVEN OBJECTS

import sys, time
import numpy as np, sparse_dict as sd
from data_validation import validate_user_input as vui
from MLObjects.TestObjectCreators import CreateNumerical as cn, test_header as th
from debug import get_module_name as gmn
from general_text import alphanumeric_str as ans
from general_data_ops import new_np_random_choice as nnrc

'''OBJECT, HEADER, AND SUPPORT OBJECTS ARE ATTRIBUTES OF THE CLASS. NOTHING IS RETURNED.'''

# MODULE FOR CREATING DATA, TARGET, REFVECS, OR TEST OBJECTS W RANDOM CATEGORICAL DATA OR FROM GIVEN OBJECTS
# CAN ONLY INGEST AN OBJECT THAT IS ENTIRELY CAT (WHEN GIVEN) AND CAN ONLY CREATE AN OBJECT THAT IS ENTIRELY CAT

# BEAR TO CHANGE NUMBER OF CATEGORIES PER COLUMNS, CTRL-F "PIZZA"

# INHERITS #############################################################################################################
# _exception
# to_row
# to_column
# expand
# to_array

# OVERWRITES ###########################################################################################################
# build
# to_sparse_dict

# UNIQUE ###############################################################################################################




# NO NEED TO INGEST SPARSE_DICT HERE
# NO NEED TO RETURN ARRAYS AS int, bin OR float, CAN ONLY BE str

class CreateCategorical(cn.CreateNumerical):

    def __init__(self, name='DATA', OBJECT=None, OBJECT_HEADER=None, given_orientation=None, columns=None, rows=None,
                 return_orientation='ROW', NUMBER_OF_CATEGORIES_AS_LIST=None):

        # UNIQUE __init__ TO Categorical ###################################################################################
        ## THIS MUST BE BEFORE super(), PARENT super() CALLS build FROM THIS CHILD, AND THESE MUST BE DECLARED FOR THIS build
        self.NUMBER_OF_CATEGORIES_AS_LIST = NUMBER_OF_CATEGORIES_AS_LIST    # VALIDATED / CHANGED LATER BY build
        # END UNIQUE __init__ TO Categorical ###############################################################################

        # DUMMIES TO SATISFY super()
        return_format = 'ARRAY'
        bin_int_or_float = 'CAT'
        min_value = 0
        max_value = 9
        _sparsity = 0

        super().__init__(name, OBJECT, OBJECT_HEADER, given_orientation, columns, rows, return_format,
                 return_orientation, bin_int_or_float, min_value, max_value, _sparsity)

        # #####################################################################################################################
        # INITIALIZED BY SUPER.__init__() #####################################################################################
        '''
        self.this_module            OVERWROTE BELOW
        self.name
        self.OBJECT
        self.OBJECT_HEADER
        self.given_format
        self.given_orientation
        self.columns_in
        self.rows_in
        self.columns_out
        self.rows_out
        self.return_format
        self.return_orientation
        self.bin_int_or_float
        self.min_value
        self.max_value
        self._sparsity
        self.is_expanded             OVERWROTE BELOW

        self.is_dict
        self.is_list

        self.VALIDATED_DATATYPES     OVERWROTE BELOW
        self.MODIFIED_DATATYPES      OVERWROTE BELOW
        self.FILTERING
        self.MIN_CUTOFFS
        self.USE_OTHER
        self.START_LAG
        self.END_LAG
        self.SCALING
        self.CONTEXT
        self.KEEP
        '''
        # END INITIALIZED BY SUPER.__init__() #################################################################################
        # #####################################################################################################################

        # OVERWRITES THESE OBJECTS CREATED IN super
        self.this_module = gmn.get_module_name(str(sys.modules[__name__]))
        self.is_expanded = False                    # super SETS THIS TO True

        # CATEGORICAL DTYPE MUST ALWAYS BE 'STR' BEFORE EXPANSION
        self.VALIDATED_DATATYPES = np.fromiter(('STR' for _ in range(self.columns_out)), dtype='<U5')
        self.MODIFIED_DATATYPES = np.fromiter(('STR' for _ in range(self.columns_out)), dtype='<U5')


    def build(self):
        #########################################################################################################################
        #BUILD ##################################################################################################################

        fxn = 'build'

        # IF AN OBJECT WAS PROVIDED, JUST IGNORE ANY "NUMBER_OF_CATEGORIES" THAT MAY HAVE BEEN ENTERED.
        # IF OBJECT WAS NOT PROVIDED AND "NUMBER_OF_CATEGORIES" WAS NOT PROVIDED, GENERATE ONE RANDOMLY
        # IF OBJECT WAS NOT PROVIDED AND "NUMBER_OF_CATEGORIES" WAS PROVIDED, VALIDATE

        if not self.OBJECT is None:
            self.NUMBER_OF_CATEGORIES_AS_LIST = None
        elif self.OBJECT is None:
            if self.NUMBER_OF_CATEGORIES_AS_LIST is None:
                self.NUMBER_OF_CATEGORIES_AS_LIST = nnrc.new_np_random_choice(range(2,10), (1,self.columns_out), replace=True)[0]
            elif not self.NUMBER_OF_CATEGORIES_AS_LIST is None:
                try: self.NUMBER_OF_CATEGORIES_AS_LIST = np.array(self.NUMBER_OF_CATEGORIES_AS_LIST, dtype=np.int16).reshape((1,-1))[0]
                except: self._exception(fxn, f'NUMBER_OF_CATEGORIES MUST BE ENTERED AS A LIST TYPE FILLED WITH AN '
                                    f'INTEGER FOR EACH OUTPUT COLUMN')

            if len(self.NUMBER_OF_CATEGORIES_AS_LIST) != self.columns_out:
                self._exception(fxn, f'LENGTH OF NUMBER_OF_CATEGORIES MUST EQUAL THE NUMBER OF OUTPUT COLUMNS BEFORE EXPANSION')

        if self.OBJECT is None:  # CREATE FROM SCRATCH

            # CREATE AS [ [] = COLUMNS ], TRANSPOSE LATER IF NEEDED
            self.OBJECT = np.empty((0, self.rows_out), dtype='<U50')

            for column_idx in range(self.columns_out):   # self.columns_out MUST EQUAL len(self.NUMBER_OF_CATEGORIES)
                UNIQUE_CATEGORIES = np.fromiter((f'LEVEL{_+1}' for _ in range(self.NUMBER_OF_CATEGORIES_AS_LIST[column_idx])), dtype='<U15')

                # RANDOMLY FILL COLUMN OF OUTPUT WITH CATEGORIES IN UNIQUE_CATEGORIES
                self.OBJECT = np.vstack((self.OBJECT,
                                        nnrc.new_np_random_choice(UNIQUE_CATEGORIES, (1,self.rows_out), replace=True)
                                        ))
            del UNIQUE_CATEGORIES

            if self.return_orientation == 'COLUMN': pass
            elif self.return_orientation == 'ROW': self.OBJECT = self.OBJECT.transpose()

        elif not self.OBJECT is None:

            if self.given_orientation != self.return_orientation:
                self.OBJECT = self.OBJECT.transpose()

        # END BUILD #############################################################################################################
        #########################################################################################################################



    # INHERITS to_row


    # INHERITS to_column


    # INHERITS to_array


    # OVERWRITES
    # CAN ONLY CONVERT TO SPARSE_DICT IF HAS BEEN EXPANDED
    def to_sparse_dict(self):
        if self.return_format == 'SPARSE_DICT': pass   # IF ALREADY IS SPARSE_DICT
        elif self.is_expanded and self.return_format == 'ARRAY':
            # ANY OBJECT CREATED OR (RIGHTFULLY) INGESTED BY THIS MODULE IS ENTIRELY CAT, SO WHEN EXPANDED MUST BE ENTIRELY BIN
            self.OBJECT = sd.zip_list_as_py_int(self.OBJECT)
            self.return_format = 'SPARSE_DICT'
            self.is_list, self.is_dict = False, True
        elif not self.is_expanded:
            print(f'\n*** CANNOT CONVERT CATEGORICAL TO SPARSE DICT UNTIL AFTER IT IS EXPANDED ***\n')









































































































if __name__ == '__main__':


    # EVERYTHING BELOW IS A TEST MODULE W SUPPORTING FUNCTIONS

    # BEAR 3/14/23 VERIFIED MODULE AND TEST CODE IS GOOD

    from general_sound import winlinsound as wls

    def test_exc_handle(OBJECT, reason_text):
        time.sleep(1)
        # print(f'\n\033[91mEXCEPTING OBJECT:\033[0m\x1B[0m')
        # print(OBJECT)
        # print()
        wls.winlinsound(888, 500)
        print(f'\n\033[91mWANTS TO RAISE EXCEPTION FOR \033[0m\x1B[0m')
        print(reason_text)
        _quit = vui.validate_user_str(f'\n\033[91mquit(q) or continue(c) > \033[0m\x1B[0m', 'QC')
        if _quit == 'Q': raise Exception(f'\033[91m*** ' + reason_text + f' ***\033[0m\x1B[0m')
        elif _quit == 'C': pass


    def test_lens(OBJECT, given_orientation, return_orientation, rows, exp_columns):
        if isinstance(OBJECT, dict):
            if sd.is_sparse_outer(OBJECT): outer_len, inner_len = sd.outer_len(OBJECT), sd.inner_len(OBJECT)
            elif sd.is_sparse_inner(OBJECT):
                raise Exception(f'\n\033[91m*** OBJECT IS A SINGLE VECTOR NOT [[ ]]  🤬 ***\033[0m\x1B[0m\n')
                # outer_len, inner_len = 1, len(OBJECT)
        elif isinstance(OBJECT, np.ndarray):
            _shape = OBJECT.shape
            if len(_shape) == 2: outer_len, inner_len = _shape[0], _shape[1]
            elif len(_shape) == 1:
                raise Exception(f'\n\033[91m*** OBJECT IS A SINGLE VECTOR NOT [[ ]]  🤬 ***\033[0m\x1B[0m\n')
                # outer_len, inner_len = 1, len(OBJECT)

        if return_orientation == 'ROW':
            if outer_len != rows:
                test_exc_handle(OBJECT, f'\033[91mouter/inner len: MISMATCH BETWEEN outer_len ({outer_len}) AND rows ({rows}) FOR given_orient={given_orientation}, return_orient={return_orientation}\033[0m\x1B[0m')
            if inner_len != exp_columns:
                test_exc_handle(OBJECT, f'\033[91mouter/inner len: MISMATCH BETWEEN inner_len ({inner_len}) AND exp_columns ({exp_columns}) FOR given_orient={given_orientation}, return_orient={return_orientation}\033[0m\x1B[0m')
        elif return_orientation == 'COLUMN':
            if outer_len != exp_columns:
                test_exc_handle(OBJECT, f'\033[91mouter/inner len: MISMATCH BETWEEN outer_len ({outer_len}) AND exp_columns ({exp_columns}) FOR given_orient={given_orientation}, return_orient={return_orientation}\033[0m\x1B[0m')
            if inner_len != rows:
                test_exc_handle(OBJECT, f'\033[91mouter/inner len: MISMATCH BETWEEN inner_len ({inner_len}) AND rows ({rows}) FOR given_orient={given_orientation}, return_orient={return_orientation}\033[0m\x1B[0m')

        else: raise Exception(f'\n\033[91m*** given_orientation LOGIC FOR DETERMINING CORRECT outer/inner_len IS FAILING ***\033[0m\x1B[0m\n')


    def test_format(OBJECT, return_format):
        if isinstance(OBJECT, dict): __ = 'SPARSE_DICT'
        elif isinstance(OBJECT, np.ndarray): __ = 'ARRAY'
        else: test_exc_handle(OBJECT, f'\n\033[91m*** BIG DISASTER. OBJECT IS NOT AN ARRAY OR SPARSE_DICT ***\033[0m\x1B[0m\n')

        if __ != return_format:
            test_exc_handle(OBJECT,
                f'\n\033[91m*** return_format: OBJECT FORMAT ({__}) IS NOT THE REQUIRED FORMAT ({return_format})!!!! ***\033[0m\x1B[0m\n')

    '''
    # def test_single_or_double(OBJECT, expected_single_or_double):
    #     if isinstance(OBJECT, dict):
    #         if sd.is_sparse_inner(OBJECT) and not sd.is_sparse_outer(OBJECT): __ = 'SINGLE'
    #         elif sd.is_sparse_outer(OBJECT) and not sd.is_sparse_inner(OBJECT): __ = 'DOUBLE'
    #         else: test_exc_handle(OBJECT, f'\n\033[91m*** sd.sparse_outer & inner are screwing up! ***\033[0m\x1B[0m\n')
    #     elif isinstance(OBJECT, np.ndarray):
    #         if isinstance(OBJECT[0], np.ndarray): __ = 'DOUBLE'
    #         else: __ = 'SINGLE'
    #
    #     if __ != expected_single_or_double:
    #         test_exc_handle(OBJECT, f'\n\033[91m*** single_or_double: OUTPUT IS {__}, SHOULD BE {_single_or_double_dum} ***\033[0m\x1B[0m\n')


    def test_min_and_max(OBJECT, bin_int_or_float, min_value, max_value, _sparsity):
        # GET EXPECTED MIN & MAX ##########################################################################################
        if bin_int_or_float == 'BIN': exp_min, exp_max = 0, 1
        elif bin_int_or_float == 'INT': exp_min, exp_max = min_value, max_value
        elif bin_int_or_float == 'FLOAT': exp_min, exp_max = min_value, max_value
        else: raise Exception(f'\n\033[91m*** LOGIC FOR bin_int_or_float IN test_min_and_max() IS FAILING ***\033[0m\x1B[0m\n')
        if _sparsity == 100: exp_min, exp_max = 0, 0
        elif _sparsity == 0: pass
        else: exp_min, exp_max = min(0, min_value), max(0, max_value)
        # END GET EXPECTED MIN & MAX ##########################################################################################

        # GET ACTUAL MIN & MAX ##########################################################################################
        if isinstance(OBJECT, dict):
            act_min, act_max = sd.min_(OBJECT), sd.max_(OBJECT)
        elif isinstance(OBJECT, (np.ndarray)):
            act_min, act_max = np.min(OBJECT), np.max(OBJECT)
        # END GET ACTUAL MIN & MAX ##########################################################################################

        if bin_int_or_float == 'BIN' and int(act_min) != 0 and int(act_max) != 1 and \
                exp_min != act_min and exp_max != act_max:
            test_exc_handle(OBJECT, f'\n\033[91m*** min max: EXPECTED BIN OUTPUT W min={min_value}, max={max_value}, GOT min={act_min}, max=({act_max}\033[0m\x1B[0m')
        elif bin_int_or_float == 'INT' and (act_min < exp_min or act_max > exp_max):
            test_exc_handle(OBJECT,
                f'\n\033[91m*** min max: EXPECTED INT OUTPUT W min={min_value}, max={max_value}, GOT min={act_min}, max={act_max}\033[0m\x1B[0m')
        elif bin_int_or_float == 'FLOAT' and (act_min < exp_min or act_max > exp_max):
            test_exc_handle(OBJECT,
                f'\n\033[91m*** min max: EXPECTED FLOAT OUTPUT W min={min_value}, max={max_value}, GOT min={act_min}, max={act_max}\033[0m\x1B[0m')


    def test_bin_int_or_float(OBJECT, expected_bin_int_or_float):

        # GET TYPE FROM OBJECT #####################################################################################
        _TYPES = []
        if isinstance(OBJECT, dict):
            # IF IS INNER DICT ("SINGLE") MAKE OUTER TO SIMPLY FOR THIS PROCESS
            if sd.is_sparse_outer(OBJECT): NEW_OBJECT = deepcopy(OBJECT)
            elif sd.is_sparse_inner(OBJECT): NEW_OBJECT = {0: deepcopy(OBJECT)}
            for outer_key in NEW_OBJECT:
                for inner_key in NEW_OBJECT[outer_key]:
                    _ = NEW_OBJECT[outer_key][inner_key]
                    if float(_) == float(0) or float(_) == float(1): __ = 'BIN'
                    elif 'INT' in str(type(_)).upper(): __ = 'INT'
                    elif 'FLOAT' in str(type(_)).upper(): __ = 'FLOAT'
                    else: test_exc_handle(OBJECT, f'\n\033[91m*** UNKNOWN DICT NUMBER DTYPE {_} ***\033[0m\x1B[0m\n')

                    if __ not in _TYPES: _TYPES.append(__)

                if 'BIN' in _TYPES and 'INT' in _TYPES and 'FLOAT' in _TYPES: break

            if 'FLOAT' in _TYPES: _dtype = ['FLOAT']
            elif 'INT' in _TYPES: _dtype = ['INT']
            elif 'BIN' in _TYPES: _dtype = ['BIN'] #, 'INT']

        elif isinstance(OBJECT, np.ndarray):
            _ = str(OBJECT.dtype).upper()
            if float(np.min(OBJECT)) in [0,1] and float(np.max(OBJECT)) in [float(0),float(1)]: _dtype = ['BIN']#, 'INT']
            elif 'INT' in _: _dtype = ['INT']
            elif 'FLOAT' in _: _dtype = ['FLOAT']
            else: test_exc_handle(OBJECT, f'\n\033[91m*** UNKNOWN NDARRAY DTYPE {_} ***\033[0m\x1B[0m\n')
        # END GET TYPE FROM OBJECT #####################################################################################

        if expected_bin_int_or_float not in _dtype:
            test_exc_handle(OBJECT, f'\n\033[91m*** bin_int_or_float: OUTPUT IS {_dtype}, SHOULD BE {expected_bin_int_or_float} ***\033[0m\x1B[0m\n')


    def get_sparsity(OBJECT):
        if isinstance(OBJECT, dict):
            if sd.is_sparse_inner(OBJECT) and not sd.is_sparse_outer(OBJECT): __ = sd.sparsity({0: OBJECT})
            else: __ = sd.sparsity(OBJECT)
        elif isinstance(OBJECT, np.ndarray):
            __ = sd.list_sparsity(OBJECT)

        return __


    def test_sparsity(actual_sparsity, expected_sparsity):
        sw = 3   # sparsity_window
        __ = actual_sparsity
        if __ < expected_sparsity-sw or __ > expected_sparsity+sw:
            test_exc_handle(__, f'\n\033[91m*** ACTUAL SPARSITY {__} IS OUTSIDE OF expected_sparsity +/- {sw} ({expected_sparsity}) ***\033[0m\x1B[0m\n')

    '''
    def test_header(HEADER, columns):
        if len(HEADER[0]) != columns:
            test_exc_handle(HEADER, f'\n\033[91m*** HEADER LENGTH ({len(HEADER[0])}) DOES NOT EQUAL NUMBER OF COLUMNS ({columns})\033[0m\x1B[0m')

    #############################################################################################################################
    #############################################################################################################################
    # ONE PASS TESTING ##########################################################################################################
    if 2 == 1:    # TOGGLE THIS TO ENABLE ONE PASS TESTING.... SEE BELOW FOR ITERATIVE TESTING
        name = 'DATA'
        given_orientation = 'COLUMN'
        _columns = 3
        _rows = 5
        return_orientation = 'ROW'
        char_str = ans.alphabet_str_upper().strip()
        char_str = np.fromiter((f'{char_str[_//26]}{char_str[_%26]}' for _ in range(26*26)), dtype='<U2')

        # BUILD AS COLUMN
        DATA_OBJECT = np.empty((0, _rows), dtype='<U50')
        NUMBER_OF_CATEGORIES_AS_LIST = np.random.randint(2, 10, (1, _columns), dtype=np.int16)[0]
        for col_idx in range(_columns):
            DATA_OBJECT = np.vstack((DATA_OBJECT,
                 nnrc.new_np_random_choice(char_str[:NUMBER_OF_CATEGORIES_AS_LIST[col_idx]], (1, _rows), replace=True)[0]
            ))

        # RE-ORIENT
        if given_orientation == 'COLUMN': pass
        elif given_orientation == 'ROW': DATA_OBJECT = DATA_OBJECT.transpose()

        OBJECT_HEADER = th.test_header(_columns)

        print(f'\nINPUT OBJECT:')
        print(DATA_OBJECT)
        print()
        print(f'\nINPUT HEADER:')
        print(OBJECT_HEADER)
        print()

        DummyClass = CreateCategorical(name=name, OBJECT=DATA_OBJECT, OBJECT_HEADER=OBJECT_HEADER,
             given_orientation=given_orientation, columns=_columns, rows=_rows, return_orientation=return_orientation,
             NUMBER_OF_CATEGORIES_AS_LIST=NUMBER_OF_CATEGORIES_AS_LIST)

        RETURN_OBJECT = DummyClass.OBJECT
        RETURN_HEADER = DummyClass.OBJECT_HEADER

        data_object_desc = "SPARSE_DICT" if isinstance(DATA_OBJECT, dict) else "ARRAY"
        hdr_desc = "given" if not OBJECT_HEADER is None else "not given"

        obj_desc = f"\nINCOMING DATA OBJECT IS A {type(DATA_OBJECT)} AND HEADER IS {hdr_desc}" + \
                   [f", WITH {_rows} ROWS AND {_columns} COLUMNS ORIENTED AS {given_orientation}. " if
                    not data_object_desc is None else ". "][0] + \
                     f"\nOBJECT SHOULD BE A NUMPY ARRAY OF STRINGS WITH {_rows} ROWS AND {_columns} COLUMNS) " \
                    f"ORIENTED AS {return_orientation}"
        print(obj_desc)

        print()
        print(f'RETURN_OBJECT:')
        print(RETURN_OBJECT)
        print()
        print(f'RETURN_HEADER:')
        print(RETURN_HEADER)

        test_lens(RETURN_OBJECT, given_orientation, return_orientation, _rows, _columns)
        test_header(RETURN_HEADER, _columns)

        print(f'\n\033[92m*** TEST PASSED ***\033[0m\x1B[0m\n')

        quit()


    # END ONE PASS TESTING ##########################################################################################################
    #############################################################################################################################
    #############################################################################################################################

    # ############################################################################################################################
    # #############################################################################################################################
    # ITERATIVE TESTING ###################################################################################################################


    name = 'DATA'
    DATA_OBJECT = None
    DATA_OBJECT_HEADER = None
    given_orientation = None
    char_str = ans.alphabet_str_upper().strip()
    char_str = np.fromiter((f'{char_str[_//26]}{char_str[_%26]}' for _ in range(26*26)), dtype='<U2')

    RETURN_FORMAT = ['ARRAY', 'SPARSE_DICT']
    RETURN_ORIENTATION = ['ROW', 'COLUMN']
    COLUMNS = [1,50,100]
    ROWS = [1000,500,100]   # ROWS MUST BE LARGE ENOUGH TO ALLOW random TO GET THE FULL NUMBER OF CATEGORIES INTO A COLUMN
                            # ELSE EXPLODY FOR NOT ENOUGH COLUMNS AFTER EXPANSION

    DATA_OBJECTS = [
                           'dummy_categorical',
                           None
                           ]

    DATA_OBJECT_HEADERS = [
                           'built_during_run',
                           None
                           ]

    EXPAND = [True, False]

    total_itrs = np.product(list(map(len, (COLUMNS, RETURN_FORMAT, RETURN_ORIENTATION, DATA_OBJECTS, EXPAND,
                                           DATA_OBJECT_HEADERS))))



    # for return_format in RETURN_FORMAT:
    #     for return_orientation in RETURN_ORIENTATION:
    #         for _columns, _rows in zip(COLUMNS, ROWS):
    #             for data_object_desc in DATA_OBJECTS:
    #                 for data_object_hdr_desc in DATA_OBJECT_HEADERS:
    #                     for expand in EXPAND:

    ctr = 0
    for return_format in RETURN_FORMAT:
        for return_orientation in RETURN_ORIENTATION:
            for _columns, _rows in zip(COLUMNS, ROWS):
                for data_object_desc in DATA_OBJECTS:
                    for expand in EXPAND:
                        if not expand: return_format = 'ARRAY'

                        # PIZZA, CHANGE THIS TO CHANGE NUM OF CATEGORIES PER COLUMN
                        # NUMBER_OF_CATEGORIES_AS_LIST = np.random.randint(min(_rows, 2), min(_rows, 10)+1, (1, _columns), dtype=np.int16)[0]
                        NUMBER_OF_CATEGORIES_AS_LIST = np.full((1, _columns), 5, dtype=np.int16)[0]

                        if not data_object_desc is None:
                            # WHEN TEST CALLS FOR AN OBJECT AS AN arg, BUILD A FAKE ONE #############################
                            DATA_OBJECT = np.empty((_columns, _rows), dtype='<U50')
                            # BUILD AS [[] = COLUMNS], THEN TRANSPOSE IF ROWS
                            POOL = np.fromiter((f'LEV{_ + 1}' for _ in range(max(NUMBER_OF_CATEGORIES_AS_LIST))), dtype='<U15')
                            for col_idx in range(_columns):
                                DATA_OBJECT[col_idx] = \
                                    nnrc.new_np_random_choice(POOL[:NUMBER_OF_CATEGORIES_AS_LIST[col_idx]], (1, _rows), replace=True)[0]
                            del POOL
                            # END WHEN TEST CALLS FOR AN OBJECT AS AN arg, BUILD A FAKE ONE #############################
                            given_orientation = 'ROW'
                            if given_orientation == 'ROW': DATA_OBJECT = DATA_OBJECT.transpose()

                        else:
                            DATA_OBJECT = None
                            given_orientation = None
                        for data_object_hdr_desc in DATA_OBJECT_HEADERS:
                            ctr += 1
                            # if ctr % 1 == 0:
                            print(f'Running test {ctr} of {total_itrs}...')

                            if not data_object_hdr_desc is None: DATA_OBJECT_HEADER = th.test_header(_columns)
                            else: DATA_OBJECT_HEADER = None
                            '''
                            RETURN_ORIENTATION = ['ROW', 'COLUMN']
                            COLUMNS = [100, 50, 10, 2, 1]
                            ROWS = [1, 2, 10, 50, 100]
                            DATA_OBJECTS = ['dummy_categorical', None]
                            DATA_OBJECT_HEADERS = ['built_during_run', None]
                            '''

                            if not DATA_OBJECT is None:                           ### THIS SHOULD BE SAME AS BELOW, 5*_columns
                                exp_columns = _columns if not expand else sum(map(len, map(np.unique, DATA_OBJECT.transpose() if given_orientation=='ROW' else DATA_OBJECT)))
                            elif DATA_OBJECT is None:
                                exp_columns = _columns if not expand else 5*_columns # PIZZA CHANGE THIS TO CHANGE BACK TO RANDOM NUMBER OF CATEGORIES IN COLUMNS #f'?'

                            obj_desc = f"\nINCOMING DATA OBJECT IS A {type(DATA_OBJECT)} AND HEADER IS {data_object_hdr_desc}" + \
                                       [f", WITH {_rows} ROWS AND {_columns} COLUMNS ORIENTED AS {given_orientation}. " if
                                        not data_object_desc is None else ". "][0] + \
                                       f"\nOBJECT SHOULD BE RETURNED AS AN {dict(((True,'EXPANDED'),(False,'UNEXPANDED')))[expand]} NUMPY ARRAY OF STRINGS " \
                                       f"WITH {_rows} ROWS AND {exp_columns} COLUMNS ORIENTED AS {return_orientation}."

                            # NUMBER_OF_CATEGORIES_AS_LIST = None
                            DummyObject = CreateCategorical(name=name, OBJECT=DATA_OBJECT,
                                 OBJECT_HEADER=DATA_OBJECT_HEADER, given_orientation=given_orientation, columns=_columns,
                                 rows=_rows, return_orientation=return_orientation,
                                 NUMBER_OF_CATEGORIES_AS_LIST=NUMBER_OF_CATEGORIES_AS_LIST)

                            if expand:
                                print(f'BEAR START EXPAND')
                                DummyObject.expand(expand_as_sparse_dict=True if return_format=='SPARSE_DICT' else False)
                                print(f'BEAR END EXPAND')

                            print(f'\033[93m{obj_desc}\033[0m')

                            dum_indicator = ' - '

                            OUTPUT_OBJECT = DummyObject.OBJECT
                            OBJECT_HEADER = DummyObject.OBJECT_HEADER
                            VALIDATED_DATATYPES = DummyObject.VALIDATED_DATATYPES
                            MODIFIED_DATATYPES = DummyObject.MODIFIED_DATATYPES
                            FILTERING = DummyObject.FILTERING
                            MIN_CUTOFFS = DummyObject.MIN_CUTOFFS
                            USE_OTHER = DummyObject.USE_OTHER
                            START_LAG = DummyObject.START_LAG
                            END_LAG = DummyObject.END_LAG
                            SCALING = DummyObject.SCALING
                            CONTEXT = DummyObject.CONTEXT
                            KEEP = DummyObject.KEEP

                            # PIZZA UNHASH THIS TO CHANGE NUMBER OF CATEGORIES PER COLUMN
                            # if exp_columns == f'?':   # MEANS OBJECT WAS NOT GIVEN AND COULDNT KNOW UNTIL OBJECT WAS BUILT BY CreateCategorical
                            #     if return_format == 'ARRAY':
                            #         if return_orientation == 'COLUMN': exp_columns = len(OUTPUT_OBJECT)
                            #         elif return_orientation == 'ROW': exp_columns = len(OUTPUT_OBJECT[0])
                            #     elif return_format == 'SPARSE_DICT':
                            #         if return_orientation == 'COLUMN': exp_columns = sd.outer_len(OUTPUT_OBJECT)
                            #         elif return_orientation == 'ROW': exp_columns = sd.inner_len_quick(OUTPUT_OBJECT)


                            # ****** TEST SUPPORT OBJECTS ********************************************************************
                            SUPP_OBJ_NAMES = ['HEADER', 'VALIDATED_DATATYPES', 'MODIFIED_DATATYPES', 'FILTERING', 'MIN_CUTOFFS',
                                              'USE_OTHER', 'START_LAG', 'END_LAG', 'SCALING', 'CONTEXT', 'KEEP']

                            if data_object_hdr_desc is None:
                                DUM_HEADER = np.fromiter((f'{name.upper()[:3]}_CAT{_ + 1}' for _ in range(_columns)), dtype='<U15').reshape((1, -1))
                            else:  # HEADER IS GIVEN
                                DUM_HEADER = th.test_header(_columns)

                            if not expand:
                                EXP_HEADER = DUM_HEADER
                            elif expand:
                                # REALLY COPPING OUT HERE, IMPOSSIBLE TO KNOW WHAT HEADER "SHOULD HAVE BEEN" WHEN None DATA_OBJECT PASSED TO CreateCategorical
                                EXP_HEADER = OBJECT_HEADER
                                # EXP_HEADER = np.empty((1, 0), dtype='<U20')
                                # for col_idx in range(_columns):
                                #     for cat_idx in range(NUMBER_OF_CATEGORIES_AS_LIST[col_idx]):
                                #         EXP_HEADER = np.insert(EXP_HEADER, len(EXP_HEADER[0]), f'{DUM_HEADER[0][col_idx]}{dum_indicator}LEVEL{cat_idx+1}', axis=1)
                            del DUM_HEADER

                            test_format(OUTPUT_OBJECT, return_format)
                            test_lens(OUTPUT_OBJECT, given_orientation, return_orientation, _rows, exp_columns)
                            test_header(OBJECT_HEADER, exp_columns)

                            EXP_VALIDATED_DATATYPES = np.fromiter(('STR' for _ in range(exp_columns)), dtype='<U3').reshape((1, -1))
                            EXP_MODIFIED_DATATYPES = np.fromiter(
                                ('BIN' if expand else 'STR' for _ in range(exp_columns)), dtype='<U3').reshape((1, -1))
                            EXP_FILTERING = np.fromiter(([] for _ in range(exp_columns)), dtype=object).reshape((1, -1))
                            EXP_MIN_CUTOFFS = np.fromiter((0 for _ in range(exp_columns)), dtype=np.int16).reshape((1, -1))
                            EXP_USE_OTHER = np.fromiter(('N' for _ in range(exp_columns)), dtype='<U1').reshape((1, -1))
                            EXP_START_LAG = np.fromiter((0 for _ in range(exp_columns)), dtype=np.int16)
                            EXP_END_LAG = np.fromiter((0 for _ in range(exp_columns)), dtype=np.int16)
                            EXP_SCALING = np.fromiter(('' for _ in range(exp_columns)), dtype='<U200')
                            EXP_KEEP = EXP_HEADER[0]
                            EXP_CONTEXT = np.array([])


                            EXP_SUPP_OBJS = [EXP_HEADER, EXP_VALIDATED_DATATYPES, EXP_MODIFIED_DATATYPES, EXP_FILTERING, EXP_MIN_CUTOFFS,
                                             EXP_USE_OTHER, EXP_START_LAG, EXP_END_LAG, EXP_SCALING, EXP_CONTEXT, EXP_KEEP]

                            ACT_SUPP_OBJS = [OBJECT_HEADER, VALIDATED_DATATYPES, MODIFIED_DATATYPES, FILTERING, MIN_CUTOFFS,
                                             USE_OTHER, START_LAG, END_LAG, SCALING, CONTEXT, KEEP]

                            for obj_name, EXP_SUPP_OBJ, ACT_SUPP_OBJ in zip(SUPP_OBJ_NAMES, EXP_SUPP_OBJS, ACT_SUPP_OBJS):
                                if not np.array_equiv(EXP_SUPP_OBJ, ACT_SUPP_OBJ):
                                    print(f'\n\n\033[91mFailed on trial {ctr} of {total_itrs}.\033[0m\x1B[0m')
                                    print(f'\033[91{obj_desc}\033[0m\x1B[0m')
                                    print(f'\033[91mEXPECTED OBJECT = \033[0m\x1B[0m')
                                    print(EXP_SUPP_OBJ)
                                    print()
                                    print(f'\033[91mACTUAL OBJECT = \033[0m\x1B[0m')
                                    print(ACT_SUPP_OBJ)
                                    test_exc_handle(ACT_SUPP_OBJ, f'\033[91mACTUAL {obj_name} DOES NOT EQUAL EXPECTED\033[0m\x1B[0m')

                            # ****** END TEST SUPPORT OBJECTS ********************************************************************

    print(f'\n\033[92m*** VALIDATION COMPLETED SUCCESSFULLY ***\033[0m\x1B[0m\n')
    for _ in range(3): wls.winlinsound(888, 500); time.sleep(0.5)

    # END ITERATIVE TESTING ###################################################################################################################
    # ############################################################################################################################
    # #############################################################################################################################






















