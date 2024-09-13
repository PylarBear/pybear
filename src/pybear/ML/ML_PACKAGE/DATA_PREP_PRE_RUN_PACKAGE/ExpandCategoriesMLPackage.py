import numpy as np, sparse_dict as sd
import sys, inspect, time
from copy import deepcopy
from debug import get_module_name as gmn
from data_validation import arg_kwarg_validater as akv
from ML_PACKAGE._data_validation import list_dict_validater as ldv
from MLObjects import ExpandCategoriesTemplate as ect
from MLObjects.SupportObjects import master_support_object_dict as msod, FullSupportObjectSplitter as fsos




class ExpandCategoriesMLPackage(ect.ExpandCategoriesTemplate):
    '''Processed as [] = columns. Expands categorical columns and associated support objects for MLPackage.
        Requires at least modified datatypes (to know what columns to ignore/expand.) To willy-nilly
        expand any old data object as simply categorical, use general_data_ops.get_dummies.'''

    def __init__(self,
                 DATA_OBJECT,
                 data_given_orientation,   # ['ROW','COLUMN']
                 data_return_orientation,   # ['ROW','COLUMN']  SHOULD ALWAYS BE GIVEN AS 'COLUMN' IN MLPackage, SO 'AS_GIVEN' IS POINTLESS
                 data_return_format,        # ['ARRAY','SPARSE_DICT']  WOULD ALWAYS BE GIVEN AS ARRAY IN MLPackage, SO 'AS_GIVEN' IS POINTLESS
                 FULL_SUPPORT_OBJECT=None,
                 CONTEXT=None,
                 KEEP=None,

                 TARGET=None,   # MUST HAVE A TARGET TO DO FULL cycler, OTHERWISE CAN ONLY GET determ!!!
                 target_given_orientation=None,
                 TARGET_TRANSPOSE=None,
                 target_transpose_given_orientation=None,
                 TARGET_AS_LIST=None,
                 target_as_list_given_orientation=None,
                 target_is_multiclass=None,
                 address_multicolinearity='PROMPT',
                 auto_drop_rightmost_column=False,
                 multicolinearity_cycler=True,
                 append_ones_for_cycler=True,
                 # THIS WILL ALLOW IF INCOMING MODIFIED_DATATYPES 'STR' CATEGORIES NOT SET UP TO BE SET TO 'NNLM50', ETC BEFOREHAND
                 # USER CAN SET THEM NOW. (BEAR IN THE FUTURE, IF INCOMING CAN BE SET TO NNLM50, ETC BEFOREHAND, SET THIS TO FALSE)
                 prompt_to_edit_given_mod_dtypes=True,
                 print_notes=False,
                 bypass_validation=False,
                 prompt_user_for_accept=True,
                 calling_module=None,
                 calling_fxn=None):



        self.this_module = gmn.get_module_name(str(sys.modules[__name__]))
        self.calling_module = calling_module if not calling_module is None else self.this_module
        self.calling_fxn = calling_fxn if not calling_fxn is None else inspect.stack()[0][3]

        self.bypass_validation = akv.arg_kwarg_validater(bypass_validation, 'bypass_validation',
                                    [True, False, None], self.calling_module, self.calling_fxn, return_if_none=False)

        FULL_SUPPORT_OBJECT = ldv.list_dict_validater(FULL_SUPPORT_OBJECT, 'FULL_SUPPORT_OBJECT')[1]

        # CANNOT PASS EMPTIES IN HEADER, ApexSupportObjectHandling WILL BLOW UP
        # PASSING A SUPOBJ W/O HEADER WHEN KEEP IS AVAILABLE WOULD BE STUPID, MAKE IT THAT KEEP CAN BE MADE FROM HEADER IF KEEP
        # WAS NOT PASSED, BUT NOT THE OTHER WAY AROUND (HEADER CANNOT BE MADE FROM KEEP -- WOULD BE NICE IF THIS COULD BE DONE,
        # BUT WAY TOO MUCH CODE GYMNASTICS TO ACCOMMODATE THIS, WITH VALIDATION INVOLVED AND ALL.
        if msod.is_empty_getter(supobj_name='HEADER', SUPOBJ=FULL_SUPPORT_OBJECT, calling_module=self.calling_module,
                                calling_fxn=self.calling_fxn):

            _ranger = range(len(FULL_SUPPORT_OBJECT[0]))
            HEADER_MAKER = lambda _generator: np.fromiter(_generator, dtype='<U20')
            # IF MOD_DTYPES IS NOT EMPTY, INCORPORATE INTO DUMMY HEADERS
            if not msod.is_empty_getter(supobj_name='MODIFIEDDATATYPES', SUPOBJ=FULL_SUPPORT_OBJECT,
                                calling_module=self.calling_module, calling_fxn=self.calling_fxn):
                _HOLDER = HEADER_MAKER((f'DAT_{FULL_SUPPORT_OBJECT[msod.QUICK_POSN_DICT()["MODIFIEDDATATYPES"]][idx][:3]}{idx + 1}' for idx in _ranger))
            else:  # MOD_DTYPES IS EMPTY, JUST CREATE VANILLA HEADER
                _HOLDER = HEADER_MAKER((f'DAT_COLUMN_{idx+1}' for idx in _ranger))
            FULL_SUPPORT_OBJECT[msod.QUICK_POSN_DICT()["HEADER"]] = _HOLDER
            del _ranger, HEADER_MAKER, _HOLDER

        # fsos INITIALIZES self.SUPPORT_OBJECTS ,self.OBJECT_HEADER, self.VALIDATED_DATATYPES, self.VALIDATED_DATATYPES,
        # self.FILTERING, self.MIN_CUTOFFS, self.USE_OTHER, self.START_LAG, self.END_LAG, self.SCALING
        fsos.FullSupObjSplitter.__init__(self, FULL_SUPPORT_OBJECT, bypass_validation=self.bypass_validation)
        # BEAR 3/14/23 START_LAG, END_LAG, SCALING NOT CURRENTLY AVAILABLE / RETURNED IN PreRunExpandCategories
        # IN THE FUTURE, THE PLAN IS THAT THESE WILL BE AVAILABLE, Expand WILL BE THE VERY LAST OPERATION ON DATA


        ##############################################################################################################################
        ##############################################################################################################################
        # DATA VALIDATION 1 - len(GIVEN SUPPORT OBJECTS) MUST == len(GIVEN_DATA_OBJECT) W-R-T GIVEN ORIENTATION #####################

        if self.bypass_validation:
            self.CONTEXT = CONTEXT if not CONTEXT is None else []
            self.KEEP = KEEP if not KEEP is None else self.SUPPORT_OBJECTS[msod.QUICK_POSN_DICT()['HEADER']]

        elif not self.bypass_validation:
            # DONT VALIDATE ALL OF THIS CHILD CLASS' args/kwargs HERE, WILL BE HANDLED BY super() IF bypass_validation = False

            data_given_orientation = akv.arg_kwarg_validater(data_given_orientation, 'data_given_orientation',
                                    ['ROW', 'COLUMN'], self.calling_module, self.calling_fxn)

            if isinstance(DATA_OBJECT, (np.ndarray, list, tuple)):
                if data_given_orientation == 'ROW': _columns = len(DATA_OBJECT[0])
                elif data_given_orientation == 'COLUMN': _columns = len(DATA_OBJECT)
            elif isinstance(DATA_OBJECT, dict):
                if data_given_orientation == 'ROW': _columns = sd.inner_len_quick(DATA_OBJECT)
                elif data_given_orientation == 'COLUMN': _columns = sd.outer_len(DATA_OBJECT)

            if len(self.SUPPORT_OBJECTS[0]) != _columns:
                self._exception(f'SUPPORT OBJECT COLUMNS AND OBJECT COLUMNS ARE NOT EQUAL W-R-T OBJECT GIVEN ORIENTATION')

            if not CONTEXT is None and not isinstance(CONTEXT, (np.ndarray, list, tuple)):
                self._exception(f'PASSED CONTEXT OBJECT MUST BE A LIST-TYPE')
            else: self.CONTEXT = CONTEXT if not CONTEXT is None else []

            if not KEEP is None and not isinstance(KEEP, (np.ndarray, list, tuple)):
                self._exception(f'PASSED "KEEP" OBJECT MUST BE A LIST-TYPE')
            else: self.KEEP = KEEP if not KEEP is None else FULL_SUPPORT_OBJECT[msod.master_support_object_dict()['HEADER']['position']]

            if not np.array_equiv(self.OBJECT_HEADER[0], self.KEEP):
                self._exception(f'HEADER AND KEEP ARE NOT EQUAL')

        # END DATA VALIDATION 1 - len(GIVEN SUPPORT OBJECTS) MUST == len(GIVEN_DATA_OBJECT) W-R-T GIVEN ORIENTATION ##################
        ##############################################################################################################################
        ##############################################################################################################################

        super().__init__(DATA_OBJECT,
                         data_given_orientation=data_given_orientation,
                         data_return_orientation=data_return_orientation,
                         data_return_format=data_return_format,
                         DATA_OBJECT_HEADER=self.OBJECT_HEADER,
                         TARGET=TARGET,
                         target_given_orientation=target_given_orientation,
                         TARGET_TRANSPOSE=TARGET_TRANSPOSE,
                         target_transpose_given_orientation=target_transpose_given_orientation,
                         TARGET_AS_LIST=TARGET_AS_LIST,
                         target_as_list_given_orientation=target_as_list_given_orientation,
                         target_is_multiclass=target_is_multiclass,
                         SUPPORT_OBJECT_AS_MDTYPES_OR_FULL_SUP_OBJ=self.SUPPORT_OBJECTS,
                         address_multicolinearity=address_multicolinearity,
                         auto_drop_rightmost_column=auto_drop_rightmost_column,
                         multicolinearity_cycler=multicolinearity_cycler,
                         append_ones_for_cycler=append_ones_for_cycler,
                         prompt_to_edit_given_mod_dtypes=prompt_to_edit_given_mod_dtypes,
                         print_notes=print_notes,
                         prompt_user_for_accept=prompt_user_for_accept,
                         bypass_validation=self.bypass_validation,
                         calling_module=self.calling_module,
                         calling_fxn=self.calling_fxn
                         )


        # END init ###################################################################################################################
        ##############################################################################################################################
        ##############################################################################################################################

    '''
    def supplemental_init(self):
        # OVERWRITTEN IN CHILD
        # ONLY USED FOR SPOOFING CHILD inits IN PARENT TO TEST
        pass


    def supplemental_support_object_expand(self, col_idx):
        # REMEMBER ITERATING THRU OBJECT BACKWARDS!!!!!

        # BEAR 12/22/22 MAY HAVE TO REVISIT THIS, NOT SURE HOW VALIDATED_DATATYPES HAS BEEN EXPANDED IN THE PAST
        # THINKING IT HAS BEEN, AND HERE FOR NOW, REPLICATING ORIGINAL VALUE. I.E., 'STR' STAYS 'STR' EVEN THO M_DTYPE WENT TO 'BIN'

        for itr in range(len(self.LEVELS)-1):   # number_of_expanded-1 FOR THE 1 ORIGINAL COLUMN
            # col_idx + 1 FOR INSERTING AFTER ORIGINAL COLUMN
            if not self.VALIDATED_DATATYPES is None:
                self.VALIDATED_DATATYPES = np.insert(self.VALIDATED_DATATYPES, col_idx+1, self.VALIDATED_DATATYPES[col_idx], axis=0)
            if not self.FILTERING is None:
                self.FILTERING = np.insert(self.FILTERING, col_idx+1, self.FILTERING[col_idx], axis=0)
            if not self.MIN_CUTOFFS is None:
                self.MIN_CUTOFFS = np.insert(self.MIN_CUTOFFS, col_idx+1, self.MIN_CUTOFFS[col_idx], axis=0)
            if not self.USE_OTHER is None:
                self.USE_OTHER = np.insert(self.USE_OTHER, col_idx+1, self.USE_OTHER[col_idx], axis=0)
            if not self.START_LAG is None:
                self.START_LAG = np.insert(self.START_LAG, col_idx+1, self.START_LAG[col_idx], axis=0)
            if not self.END_LAG is None:
                self.END_LAG = np.insert(self.END_LAG, col_idx+1, self.END_LAG[col_idx], axis=0)
            if not self.SCALING is None:
                self.SCALING = np.insert(self.SCALING, col_idx+1, self.SCALING[col_idx], axis=0)
        # NOT self.KEEP
        # NOT self.CONTEXT



    def supplemental_restore(self):
        # 12/26/22 THIS IS NOT USED IN PARENT AND IS UP FOR DELETE BEAR
        # THIS CHILD MODULE DOES NOT ALLOW RESTART OF EXPANSION SO NO NEED FOR RESTORE
        pass



    def supplemental_validation(self):
        # OVERWRITTEN IN CHILD
        NAMES = ['VALIDATED_DATATYPES', 'FILTERING', 'MIN_CUTOFFS', 'USE_OTHER', 'KEEP', 'START_LAG', 'END_LAG', 'SCALING']

        OBJS = [self.VALIDATED_DATATYPES, self.FILTERING, self.MIN_CUTOFFS, self.USE_OTHER, self.KEEP, self.START_LAG,
                self.END_LAG, self.SCALING]

        _ = len(self.DATA_OBJECT)
        for obj_idx in range(len(OBJS)):
            __ = len(OBJS[obj_idx])
            if _ != __:
                self._exception(f'len OF {NAMES[obj_idx]} ({__}) DOES NOT EQUAL len OF DATA_OBJECT ({_}) '
                                f'W-R-T GIVEN DATA ORIENTATION')


    def supplemental_return(self):
        # ? 12/23/22 NOT SURE WHAT THE INTENT WAS HERE, LOOKS LIKE IT BUILDS SUPPORT OBJECTS IF THEY DONT EXIST
        if self.data_return_format == 'ARRAY':
            _outer = len(self.DATA)
            _inner = len(self.DATA[0])
        elif self.data_return_format == 'SPARSE_DICT':
            _outer = sd.outer_len(self.DATA)
            _inner = sd.inner_len_quick(self.DATA)
        if self.data_return_orientation == 'COLUMN': _columns, _rows = _outer, _inner
        elif self.data_return_orientation == 'ROW': _columns, _rows = _inner, _outer


        if self.VALIDATED_DATATYPES is None:
            self.VALIDATED_DATATYPES = deepcopy(self.MODIFIED_DATATYPES)
        if self.FILTERING is None:
            self.FILTERING = np.fromiter(('' for _ in range(_columns)), dtype='<U1')
        if self.MIN_CUTOFFS is None:
            self.MIN_CUTOFFS = np.zeros(_columns, dtype=np.int32)
        if self.USE_OTHER is None:
            self.USE_OTHER = np.fromiter(('N' for _ in range(_columns)), dtype='<U1')
        if self.CONTEXT is None:
            self.CONTEXT = []
        if self.KEEP is None:
            self.KEEP = deepcopy(self.DATA_OBJECT_HEADER[0])
        if self.START_LAG is None:
            self.START_LAG = np.zeros(_columns, dtype=np.int32)
        if self.END_LAG is None:
            self.END_LAG = np.zeros(_columns, dtype=np.int32)
        if self.SCALING is None:
            self.SCALING = np.fromiter(('-' for _ in range(_columns)), dtype='<U1')
    '''


    def context_update_for_column_drop(self, words):
        self.CONTEXT.append(words)
















if __name__ == '__main__':

    # TEST MODULE!

    # USE THIS TO TEST THE CORRECTNESS OF OUTPUT OBJECTS OVER VARIOUS INPUT SHAPES/FORMATS/ORIENTATIONS AND VARIOUS OUTPUT
    # SHAPES/FORMATS/ORIENTATIONS
    # TO TEST FUNCTIONALITY OF PROMPTS/NOTES, MULTICOLIN CYCLER USE TEST IN ExpandCategoriesTemplate

    # BEAR 3/16/23 VERIFIED THIS MODULE AND THIS TEST CODE FUNCTION CORRECTLY W/O NNLM50 (SEE NOTES BELOW)

    from general_sound import winlinsound as wls

    # 12/24/22 NNLM50 IS LINEARLY PILING UP RAM WITH EVERY ITERATION OF TEST. INTERNET SAYS THERE MAY BE SOME MEMORY LEAK BUT
    # DOES NOT GIVE A SOLUTION.  tf.keras.clear_session() DOES NOT HELP. THE ONLY SOLUTION IS TO TEST NNLM50 LESS, AND
    # SEPARATE FROM ALL THE OTHER DTYPES :(   CTRL-F TO "EGGNOG"

    dum_indicator = ' - '
    alpha_str = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

    calling_module = gmn.get_module_name(str(sys.modules[__name__]))
    calling_fxn = f'guard_test'

    # BEAR BYPASS 3/13/2023
    # embed = hub.load("https://tfhub.dev/google/nnlm-en-dim50/2")

    from general_text import Lexicon as lx
    from general_data_ops import new_np_random_choice as nnrc
    from MLObjects.SupportObjects import CompileFullSupportObject as cfso

    POOL_OF_WORDS = nnrc.new_np_random_choice(lx.Lexicon().LEXICON, (1,50), replace=False).reshape((1,-1))[0]

    POOL_OF_DTYPES = ['BIN', 'INT', 'FLOAT', 'STR', 'SPLIT_STR'] #, 'NNLM50']           # EGGNOG

    # NONE OF THE SUPPORT OBJECTS CANS BE EMPTY, THIS ExpandCategoriesTestObject REQUIRES A FULL SUPPORT OBJECT, AT ALL POINTS
    # WHEN EXPAND IS CALLED AN OBJECT AND ITS FULL SUPPORT OBJECT SHOULD (MUST) EXIST

    MASTER_BYPASS_VALIDATION = [False, True]
    MASTER_DATA_OBJECT = ['construct on the fly']       #, None]
    MASTER_SIZE = ((3,2), (3,3), (2,3))         #((30,20), (20,20), (20,30))
    MASTER_DATA_GIVEN_ORIENTATION = ['COLUMN', 'ROW']
    MASTER_DATA_RETURN_ORIENTATION = ['COLUMN', 'ROW']
    MASTER_DATA_GIVEN_FORMAT = ['ARRAY', 'SPARSE_DICT']
    MASTER_DATA_RETURN_FORMAT = ['ARRAY', 'SPARSE_DICT']
    MASTER_AUTO_DROP = [False, True]
    MASTER_VALIDATED_DATATYPES = ['construct on the fly']
    MASTER_MODIFIED_DATATYPES = ['construct on the fly']
    MASTER_FILTERING = ['construct on the fly']
    MASTER_MIN_CUTOFFS = ['construct on the fly']
    MASTER_USE_OTHER = ['construct on the fly']
    MASTER_CONTEXT = ['construct on the fly', None]
    MASTER_HEADER = ['construct on the fly', None]
    MASTER_KEEP = ['construct on the fly', None]
    MASTER_START_LAG = ['construct on the fly']
    MASTER_END_LAG = ['construct on the fly']
    MASTER_SCALING = ['construct on the fly']


    total_trials = np.product(list(map(len, [MASTER_BYPASS_VALIDATION, MASTER_DATA_OBJECT, MASTER_SIZE, MASTER_DATA_GIVEN_ORIENTATION,
                                   MASTER_DATA_RETURN_ORIENTATION, MASTER_DATA_GIVEN_FORMAT, MASTER_DATA_RETURN_FORMAT, MASTER_HEADER,
                                   MASTER_AUTO_DROP, MASTER_VALIDATED_DATATYPES, MASTER_FILTERING, MASTER_MIN_CUTOFFS,
                                   MASTER_USE_OTHER, MASTER_CONTEXT, MASTER_START_LAG, MASTER_SCALING])))

    ctr = 0
    for bypass_validation in MASTER_BYPASS_VALIDATION:
        for GIVEN_DATA_OBJECT in MASTER_DATA_OBJECT:
            for size in MASTER_SIZE:
                for data_given_format in MASTER_DATA_GIVEN_FORMAT:
                    for data_given_orientation in MASTER_DATA_GIVEN_ORIENTATION:
                        for auto_drop_rightmost_column in MASTER_AUTO_DROP:
                            for GIVEN_VALIDATED_DATATYPES, GIVEN_MODIFIED_DATATYPES, GIVEN_KEEP in zip(MASTER_VALIDATED_DATATYPES, MASTER_MODIFIED_DATATYPES, MASTER_KEEP):
                                _columns = size[1]
                                _rows = size[0]

                                # TO TEST ALL DTYPES ONE COLUMN EACH
                                # BASE_EXP_MODIFIED_DATATYPES = np.array(['BIN', 'INT', 'FLOAT', 'STR', 'SPLIT_STR', 'BIN']) #, 'NNLM50'])
                                # CREATE RAW DATA WITH RANDOM DTYPES
                                BASE_EXP_MODIFIED_DATATYPES = np.random.choice(POOL_OF_DTYPES, _columns, replace=True)
                                # TO TEST INDIVIDUAL DTYPES       # EGGNOG
                                # BASE_EXP_MODIFIED_DATATYPES = np.fromiter(('NNLM50' for _ in range(_columns)), dtype='<U10')

                                if not GIVEN_MODIFIED_DATATYPES is None:   # else MODIFIED_DATATYPES_OBJ STAYS None
                                    GIVEN_MODIFIED_DATATYPES = deepcopy(BASE_EXP_MODIFIED_DATATYPES)

                                BASE_EXP_VALIDATED_DATATYPES = np.fromiter((msod.val_reverse_lookup()[_] for _ in BASE_EXP_MODIFIED_DATATYPES), dtype='<U10')
                                if not GIVEN_VALIDATED_DATATYPES is None:
                                    GIVEN_VALIDATED_DATATYPES = deepcopy(BASE_EXP_VALIDATED_DATATYPES)


                                # BUILD BASE_EXP_DATA_OBJECT
                                BASE_EXP_DATA_OBJECT = np.empty((0,_rows), dtype=object)
                                for col_idx in range(_columns):
                                    __ = BASE_EXP_MODIFIED_DATATYPES[col_idx]
                                    if __ == 'BIN':
                                        BASE_EXP_DATA_OBJECT = np.vstack((BASE_EXP_DATA_OBJECT, np.random.randint(0,2,(1,_rows), dtype=np.int8)))
                                    elif __ == 'INT':
                                        BASE_EXP_DATA_OBJECT = np.vstack((BASE_EXP_DATA_OBJECT, np.random.randint(0,10,(1,_rows), dtype=np.int32)))
                                    elif __ == 'FLOAT':
                                        BASE_EXP_DATA_OBJECT = np.vstack((BASE_EXP_DATA_OBJECT, np.random.uniform(0,10,(1,_rows))))
                                    elif __ == 'STR':
                                        BASE_EXP_DATA_OBJECT = np.vstack((BASE_EXP_DATA_OBJECT, np.random.choice(POOL_OF_WORDS, _rows, replace=True)))
                                    elif __ == 'SPLIT_STR':
                                        HOLDER = np.empty(_rows, dtype='<U200')
                                        for row_idx in range(len(HOLDER)):
                                            HOLDER[row_idx] = ' '.join(np.random.choice(POOL_OF_WORDS, 3, replace=True).tolist())
                                        BASE_EXP_DATA_OBJECT = np.vstack((BASE_EXP_DATA_OBJECT, HOLDER))  # ITERATING FORWARD TO BUILD
                                        del HOLDER
                                    elif __=='NNLM50':
                                        HOLDER = np.empty(_rows, dtype='<U200')
                                        for row_idx in range(len(HOLDER)):
                                            HOLDER[row_idx] = f' '.join(np.random.choice(POOL_OF_WORDS, 5).tolist())
                                        BASE_EXP_DATA_OBJECT = np.vstack((BASE_EXP_DATA_OBJECT, HOLDER))  # ITERATING FORWARD TO BUILD
                                        del HOLDER
                                    else:
                                        wls.winlinsound(888, 500)
                                        raise Exception(f'BAD DTYPE "{__}" IN EXP_MOD_DTYPES DURING BASE_EXP_DATA_OBJECT BUILD')


                                GIVEN_DATA_OBJECT = deepcopy(BASE_EXP_DATA_OBJECT)  # EXP_DATA IS 'COLUMN', NEED TO BE COLUMN WHEN PROCESSED TO FINAL EXP_DATA

                                if data_given_orientation == 'COLUMN': pass
                                elif data_given_orientation == 'ROW':
                                    GIVEN_DATA_OBJECT = GIVEN_DATA_OBJECT.transpose()

                                if data_given_format == 'SPARSE_DICT':
                                    # IF BASE_DATA HAS STR CHARS IN IT, CANNOT CONVERT TO SPARSE_DICT, SO JUST LEAVE AS ARRAY
                                    try: GIVEN_DATA_OBJECT = sd.zip_list_as_py_float(GIVEN_DATA_OBJECT)
                                    except: data_given_format = 'ARRAY'

                                # EXP_DATA DOES NOT EQUAL GIVEN_DATA!  JUST MAKE A TEMPLATE TO MODIFY TO GET EXP_DATA!
                                exp_data_changed = True

                                for data_return_format in MASTER_DATA_RETURN_FORMAT:
                                    for data_return_orientation in MASTER_DATA_RETURN_ORIENTATION:
                                        for GIVEN_HEADER in MASTER_HEADER:
                                            if not GIVEN_HEADER is None:
                                                GIVEN_HEADER = np.fromiter((f'TEST_COLUMN_{_+1}' for _ in range(_columns)), dtype='<U30').reshape((1, -1))
                                                BASE_EXP_HEADER = deepcopy(GIVEN_HEADER)
                                            else:
                                                BASE_EXP_HEADER = np.fromiter((f'DAT_{GIVEN_MODIFIED_DATATYPES[_][:3]}{_+1}' for _ in range(_columns)), dtype='<U30').reshape((1, -1))
                                                # GIVEN_HEADER stays None

                                            # OUTPUT KEEP (EXP_KEEP) IS AVAILABLE IRREGARDLESS IF KEEP IS GIVEN
                                            EXP_KEEP = deepcopy(BASE_EXP_HEADER[0])
                                            if not GIVEN_KEEP is None:
                                                GIVEN_KEEP = deepcopy(BASE_EXP_HEADER[0])
                                            # else: GIVEN_KEEP WOULD STAY None


                                            for GIVEN_FILTERING in MASTER_FILTERING:
                                                BASE_EXP_FILTERING = np.empty(_columns, dtype=object)
                                                for idx in range(_columns):
                                                    _, __ = tuple(np.fromiter((_ for _ in alpha_str), dtype='<U1')[np.random.randint(0,26, (1,2))[0]])
                                                    if [f'{_}{__}'] not in BASE_EXP_FILTERING:   # MAKE ALL FILTERINGS BE UNIQUE
                                                        BASE_EXP_FILTERING[idx] = [f'{_}{__}']
                                                    else: idx -= 1
                                                if not GIVEN_FILTERING is None: GIVEN_FILTERING = deepcopy(BASE_EXP_FILTERING)
                                                # else GIVEN_FILTERING stays None

                                                BASE_EXP_MIN_CUTOFFS = np.random.randint(0,10,_columns).astype(np.int8)
                                                for GIVEN_MIN_CUTOFFS in MASTER_MIN_CUTOFFS:
                                                    if not GIVEN_MIN_CUTOFFS is None: GIVEN_MIN_CUTOFFS = deepcopy(BASE_EXP_MIN_CUTOFFS)
                                                    # else GIVEN_MIN_CUTOFFS stays None

                                                    BASE_EXP_USE_OTHER = np.fromiter((np.random.choice(['Y','N'], _columns).astype('<U1')[0] if _>0  else 'N' for _ in GIVEN_MIN_CUTOFFS), dtype='<U1')
                                                    for GIVEN_USE_OTHER in MASTER_USE_OTHER:
                                                        if not GIVEN_USE_OTHER is None: GIVEN_USE_OTHER = deepcopy(BASE_EXP_USE_OTHER)
                                                        # else GIVEN_USE_OTHER stays None

                                                        for GIVEN_CONTEXT in MASTER_CONTEXT:
                                                            if not GIVEN_CONTEXT is None:
                                                                GIVEN_CONTEXT = [f'Test context junk 1', f'Test context junk 2', f'Test context junk 3']
                                                                EXP_CONTEXT = [f'Test context junk 1', f'Test context junk 2', f'Test context junk 3']
                                                            else:
                                                                # GIVEN_CONTEXT stays None
                                                                EXP_CONTEXT = []

                                                            EXP_START_LAG = np.random.randint(0,10,_columns).astype(np.int8)
                                                            EXP_END_LAG = np.fromiter((EXP_START_LAG[_]+10 for _ in range(_columns)), dtype=np.int8)
                                                            for GIVEN_START_LAG, GIVEN_END_LAG in zip(MASTER_START_LAG, MASTER_END_LAG):
                                                                if not GIVEN_START_LAG is None: GIVEN_START_LAG = deepcopy(EXP_START_LAG)
                                                                # else GIVEN_START_LAG stays None
                                                                if not GIVEN_END_LAG is None: GIVEN_END_LAG = deepcopy(EXP_END_LAG)
                                                                # else GIVEN_END_LAG stays None

                                                                EXP_SCALING = np.fromiter((msod.empty_value() for _ in range(_columns)), dtype='<U30')
                                                                print(f'*' * 90)
                                                                for GIVEN_SCALING in MASTER_SCALING:
                                                                    if not GIVEN_SCALING is None: GIVEN_SCALING = deepcopy(EXP_SCALING)
                                                                    # else GIVEN_SCALING stays None

                                                                    ctr += 1
                                                                    print(f'Running trial {ctr} of max possible {total_trials:,}...')

                                                                    if exp_data_changed:
                                                                        exp_data_given_format = data_given_format
                                                                        exp_data_given_orientation = data_given_orientation
                                                                        exp_data_return_format = data_return_format
                                                                        exp_data_return_orientation = data_return_orientation
                                                                        exp_auto_drop_rightmost_column = auto_drop_rightmost_column
                                                                        exp_calling_module = calling_module
                                                                        exp_calling_fxn = calling_fxn

                                                                        # RESTORE OBJECTS THAT ARE MODIFIED IN-PLACE BACK TO BASE
                                                                        EXP_HEADER = deepcopy(BASE_EXP_HEADER)
                                                                        EXP_VALIDATED_DATATYPES = deepcopy(BASE_EXP_VALIDATED_DATATYPES)
                                                                        EXP_MODIFIED_DATATYPES = deepcopy(BASE_EXP_MODIFIED_DATATYPES)
                                                                        EXP_FILTERING = deepcopy(BASE_EXP_FILTERING)
                                                                        EXP_MIN_CUTOFFS = deepcopy(BASE_EXP_MIN_CUTOFFS)
                                                                        EXP_USE_OTHER = deepcopy(BASE_EXP_USE_OTHER)

                                                                        if exp_data_return_format=='ARRAY': EXP_DATA_OBJECT = np.empty((0,_rows), dtype=object)
                                                                        elif exp_data_return_format=='SPARSE_DICT': EXP_DATA_OBJECT = {}

                                                                        for idx in range(_columns-1, -1, -1):
                                                                            # RECORED CURRENT idx VALUE IN SUPPORT OBJECTS BEFORE DELETE OF THESE VALUES (SEE NEXT STEP)
                                                                            exp_header_value = EXP_HEADER[0][idx]
                                                                            exp_validated_value =EXP_VALIDATED_DATATYPES[idx]
                                                                            exp_modified_value = EXP_MODIFIED_DATATYPES[idx]
                                                                            exp_filtering_value = EXP_FILTERING[idx]
                                                                            exp_min_cutoffs_value = EXP_MIN_CUTOFFS[idx]
                                                                            exp_use_other_value = EXP_USE_OTHER[idx]
                                                                            # exp_keep_value = EXP_KEEP[idx]
                                                                            exp_start_lag_value = EXP_START_LAG[idx]
                                                                            exp_end_lag_value = EXP_END_LAG[idx]
                                                                            exp_scaling_value = EXP_SCALING[idx]

                                                                            # DELETE CURRENT idx FROM ALL SUPPORT OBJECTS HERE, TO AVOID CODE GYMNASTICS WHEN INSERTING W _replicates
                                                                            EXP_HEADER = np.delete(EXP_HEADER, idx, axis=1)
                                                                            EXP_VALIDATED_DATATYPES = np.delete(EXP_VALIDATED_DATATYPES, idx, axis=0)
                                                                            EXP_MODIFIED_DATATYPES = np.delete(EXP_MODIFIED_DATATYPES, idx, axis=0)
                                                                            EXP_FILTERING = np.delete(EXP_FILTERING, idx, axis=0)
                                                                            EXP_MIN_CUTOFFS = np.delete(EXP_MIN_CUTOFFS, idx, axis=0)
                                                                            EXP_USE_OTHER = np.delete(EXP_USE_OTHER, idx, axis=0)
                                                                            # DONT DELETE ANYTHING FROM CONTEXT
                                                                            # DONT CHANGE KEEP, IS PRE-EXPANSION HEADER
                                                                            EXP_START_LAG = np.delete(EXP_START_LAG, idx, axis=0)
                                                                            EXP_END_LAG = np.delete(EXP_END_LAG, idx, axis=0)
                                                                            EXP_SCALING = np.delete(EXP_SCALING, idx,  axis=0)

                                                                            if exp_modified_value in ['BIN', 'INT', 'FLOAT']:
                                                                                NEW_DATA_BLOCK = BASE_EXP_DATA_OBJECT[idx].copy()

                                                                                if exp_data_return_format == 'ARRAY':
                                                                                    EXP_DATA_OBJECT = np.vstack((NEW_DATA_BLOCK, EXP_DATA_OBJECT))

                                                                                elif exp_data_return_format == 'SPARSE_DICT':
                                                                                    if exp_modified_value == 'FLOAT':
                                                                                        EXP_DATA_OBJECT = sd.core_merge_outer(sd.zip_list_as_py_float(NEW_DATA_BLOCK.reshape((1,-1))), EXP_DATA_OBJECT)[0]

                                                                                    elif exp_modified_value in ['BIN', 'INT']:
                                                                                        EXP_DATA_OBJECT = sd.core_merge_outer(sd.zip_list_as_py_int(NEW_DATA_BLOCK.reshape((1, -1))), EXP_DATA_OBJECT)[0]
                                                                                del NEW_DATA_BLOCK

                                                                                _replicates = 1
                                                                                mod_dtypes_insert = exp_validated_value

                                                                                EXP_HEADER = np.hstack((
                                                                                    EXP_HEADER[..., :idx], [[exp_header_value]], EXP_HEADER[..., idx:]
                                                                                ))

                                                                            elif exp_modified_value == 'STR':
                                                                                UNIQUES = np.unique(BASE_EXP_DATA_OBJECT[idx])
                                                                                # THIS IS HANDLED DIFFERENTLY THAN SPLIT_STR AND NNLM50, THOSE BUILD AN EXPANDED BLOCK THEN USE vstack TO OVERWRITE THE
                                                                                # ORIGINAL COLUMN IN DATA; THIS BUILDDS OUT THE BIN COLUMNS DIRECTLY INTO THE DATA ON THE FLY, THE DELETES ORIGINAL COLUMN
                                                                                NEW_DATA_BLOCK = np.empty((0,_rows), dtype=np.int8)
                                                                                for unique_idx in range(len(UNIQUES)):
                                                                                    if not (auto_drop_rightmost_column and unique_idx == len(UNIQUES)-1):
                                                                                        NEW_DATA_BLOCK = np.vstack((NEW_DATA_BLOCK, np.int8(BASE_EXP_DATA_OBJECT[idx]==UNIQUES[unique_idx])))

                                                                                if exp_data_return_format=='ARRAY':
                                                                                    EXP_DATA_OBJECT = np.vstack((NEW_DATA_BLOCK, EXP_DATA_OBJECT))
                                                                                elif exp_data_return_format=='SPARSE_DICT':
                                                                                    EXP_DATA_OBJECT = sd.core_merge_outer(sd.zip_list_as_py_int(NEW_DATA_BLOCK), EXP_DATA_OBJECT)[0]

                                                                                mod_dtypes_insert = 'BIN'
                                                                                if auto_drop_rightmost_column:
                                                                                    _replicates = len(UNIQUES)-1
                                                                                    if not GIVEN_HEADER is None:
                                                                                        _insert = f'Deleted {GIVEN_HEADER[0][idx]}{dum_indicator}{UNIQUES[-1]} for multicolinearity auto_drop_rightmost_column'
                                                                                    else:
                                                                                        # 3/16/23 BELIEVING THAT THIS OPERATION CAN ONLY HAPPEN TO STR COLUMNS, NOT SPLIT_STR
                                                                                        _insert = f'Deleted DAT_STR{idx+1}{dum_indicator}{UNIQUES[-1]} for multicolinearity auto_drop_rightmost_column'
                                                                                    EXP_CONTEXT = np.hstack((EXP_CONTEXT, _insert))
                                                                                    del _insert
                                                                                else: _replicates = len(UNIQUES)

                                                                                # EXP HEADER ########################
                                                                                EXP_HEADER = np.hstack((EXP_HEADER[...,:idx],
                                                                                                        np.fromiter((f'{exp_header_value}{dum_indicator}{UNIQUES[_]}' for _ in range(_replicates)), dtype='<U100').reshape((1,-1)),
                                                                                                        EXP_HEADER[...,idx:]
                                                                                                       ))
                                                                                del UNIQUES

                                                                            elif exp_modified_value == 'SPLIT_STR':

                                                                                SPLIT_HOLDER = np.empty(_rows, dtype=object)
                                                                                UNIQUES = np.empty(0, dtype='<U30')

                                                                                for row_idx in range(len(BASE_EXP_DATA_OBJECT[idx])):
                                                                                    _SPLIT = np.char.upper(np.char.split(BASE_EXP_DATA_OBJECT[idx][row_idx]).tolist())
                                                                                    SPLIT_HOLDER[row_idx] = _SPLIT
                                                                                    UNIQUES = np.hstack((UNIQUES, np.unique(_SPLIT)))
                                                                                del _SPLIT
                                                                                UNIQUES = np.unique(UNIQUES)

                                                                                DUMMY_INSERT = np.zeros((len(UNIQUES), _rows), dtype=np.int16)
                                                                                for unique_idx in range(len(UNIQUES)):
                                                                                    for row_idx in range(_rows):
                                                                                        DUMMY_INSERT[unique_idx][row_idx] = np.sum(np.int8(SPLIT_HOLDER[row_idx]==UNIQUES[unique_idx]))

                                                                                if exp_data_return_format=='ARRAY':
                                                                                    EXP_DATA_OBJECT = np.vstack((DUMMY_INSERT, EXP_DATA_OBJECT))

                                                                                elif exp_data_return_format=='SPARSE_DICT':
                                                                                    EXP_DATA_OBJECT = sd.core_merge_outer(sd.zip_list_as_py_int(DUMMY_INSERT), EXP_DATA_OBJECT)[0]

                                                                                # BEAR 3/13/23 IDEALLY GETS OPTION FOR BIN, WHEN A SPLIT_STR COLUMN COMES OUT IN [0,1]
                                                                                mod_dtypes_insert = 'INT'

                                                                                _replicates = len(UNIQUES)

                                                                                # EXP HEADER ########################

                                                                                EXP_HEADER = np.hstack((EXP_HEADER[...,:idx],
                                                                                                        np.fromiter((f'{exp_header_value}{dum_indicator}{UNIQUES[_]}' for _ in range(_replicates)), dtype='<U100').reshape((1,-1)),
                                                                                                        EXP_HEADER[...,idx:]
                                                                                                        ))

                                                                                del SPLIT_HOLDER, UNIQUES, DUMMY_INSERT

                                                                            elif exp_modified_value == 'NNLM50':

                                                                                NNLM50 = np.array(embed(BASE_EXP_DATA_OBJECT[idx]), dtype=np.float64).transpose()

                                                                                if exp_data_return_format=='ARRAY':
                                                                                    EXP_DATA_OBJECT = np.vstack((NNLM50, EXP_DATA_OBJECT))
                                                                                elif exp_data_return_format=='SPARSE_DICT':
                                                                                    EXP_DATA_OBJECT = sd.core_merge_outer(sd.zip_list_as_py_float(NNLM50), EXP_DATA_OBJECT)[0]

                                                                                del NNLM50
                                                                                # tf.keras.backend.clear_session()

                                                                                _replicates = 50
                                                                                mod_dtypes_insert = 'FLOAT'

                                                                                # EXP HEADER ########################
                                                                                EXP_HEADER = np.hstack((EXP_HEADER[...,:idx],
                                                                                                        np.fromiter((f'{exp_header_value}_NNLM50_{_+1}' for _ in range(_replicates)), dtype='<U100').reshape((1,-1)),
                                                                                                        EXP_HEADER[...,idx:]
                                                                                                       ))
                                                                            else:
                                                                                wls.winlinsound(888,500)
                                                                                raise Exception(f'\n*** INVALID exp_modified_value "{exp_modified_value}"... EXPANSION OF RAW DATA COLUMN ILLEGALLY BYPASSED ***\n')

                                                                            # MODIFY EXP SUPPORT OBJECTS
                                                                            for rep in range(_replicates):
                                                                                # HEADER IS DONE UNDER EACH DTYPE
                                                                                EXP_MODIFIED_DATATYPES = np.insert(EXP_MODIFIED_DATATYPES, idx, mod_dtypes_insert, axis=0)
                                                                                EXP_VALIDATED_DATATYPES = np.insert(EXP_VALIDATED_DATATYPES, idx, exp_validated_value, axis=0)
                                                                                # 3/13/23 DONT USE np.insert TO DIRECTLY INSERT [exp_filtering_value], INSTEAD OF GIVING [['AA'], ['BB']] GIVES [['AA'], 'BB']
                                                                                EXP_FILTERING = np.insert(EXP_FILTERING, idx, None, axis=0)   # CREATE DUMMY POSN TO BE OVERWRIT
                                                                                EXP_FILTERING[idx] = exp_filtering_value
                                                                                EXP_MIN_CUTOFFS = np.insert(EXP_MIN_CUTOFFS, idx, exp_min_cutoffs_value, axis=0)
                                                                                EXP_USE_OTHER = np.insert(EXP_USE_OTHER, idx, exp_use_other_value, axis=0)
                                                                                # NO INSERT TO CONTEXT, CONTEXT ONLY ABOVE if auto_drop_rightmost FOR STR
                                                                                # EXP_KEEP = np.insert(EXP_KEEP, idx, exp_keep_value, axis=0)
                                                                                # DONT CHANGE KEEP!
                                                                                EXP_START_LAG = np.insert(EXP_START_LAG, idx, exp_start_lag_value, axis=0)
                                                                                EXP_END_LAG = np.insert(EXP_END_LAG, idx, exp_end_lag_value, axis=0)
                                                                                EXP_SCALING = np.insert(EXP_SCALING, idx, exp_scaling_value, axis=0)

                                                                        EXP_DATA_BACKUP = deepcopy(EXP_DATA_OBJECT) if isinstance(EXP_DATA_OBJECT, dict) else EXP_DATA_OBJECT.copy()

                                                                    elif not exp_data_changed:

                                                                        EXP_DATA_OBJECT = deepcopy(EXP_DATA_BACKUP) if isinstance(EXP_DATA_BACKUP, dict) else EXP_DATA_BACKUP.copy()

                                                                    raw_data_changed = False

                                                                    if exp_data_return_format == 'ARRAY' and isinstance(EXP_DATA_OBJECT, dict):
                                                                        EXP_DATA_OBJECT = sd.unzip_to_ndarray_float64(EXP_DATA_OBJECT)[0]
                                                                    if exp_data_return_format == 'SPARSE_DICT' and isinstance(EXP_DATA_OBJECT, np.ndarray):
                                                                        EXP_DATA_OBJECT = sd.zip_list_as_py_float(EXP_DATA_OBJECT)

                                                                    if exp_data_return_orientation == 'ROW':
                                                                        if exp_data_return_format == 'ARRAY':
                                                                            EXP_DATA_OBJECT = EXP_DATA_OBJECT.transpose()
                                                                        elif exp_data_return_format == 'SPARSE_DICT':
                                                                            EXP_DATA_OBJECT = sd.core_sparse_transpose(EXP_DATA_OBJECT)





                                                                    expected_output = (f'Expected output:\n',
                                                                       f'VAL_dtypes = {BASE_EXP_MODIFIED_DATATYPES}\n',
                                                                       f'size = {size}\n',
                                                                       f'exp_calling_module = {exp_calling_module}\n',
                                                                       f'exp_calling_fxn = {exp_calling_fxn}\n',
                                                                       f'exp_data_given_orientation = {exp_data_given_orientation}\n',
                                                                       f'exp_data_given_format = {exp_data_given_format}\n',
                                                                       f'exp_auto_drop_rightmost_column = {exp_auto_drop_rightmost_column}\n',
                                                                       f'exp_data_return_orientation = {exp_data_return_orientation}\n',
                                                                       f'exp_data_return_format = {exp_data_return_format}\n',
                                                                       f'EXP_DATA_OBJECT = {"A non-None OBJECT" if not GIVEN_DATA_OBJECT is None else None}\n',
                                                                       f'EXP_HEADER = {"A non-None OBJECT" if not GIVEN_HEADER is None else None}\n',
                                                                       f'EXP_MODIFIED_DATATYPES = {"A non-None OBJECT" if not GIVEN_MODIFIED_DATATYPES is None else None}\n',
                                                                       f'EXP_VALIDATED_DATATYPES = {"A non-None OBJECT" if not GIVEN_VALIDATED_DATATYPES is None else None}\n',
                                                                       f'EXP_FILTERING = {"A non-None OBJECT" if not GIVEN_FILTERING is None else None}\n',
                                                                       f'EXP_CONTEXT = {"A non-None OBJECT" if not GIVEN_CONTEXT is None else None}\n',
                                                                       f'EXP_KEEP = {"A non-None OBJECT" if not GIVEN_KEEP is None else None}\n',
                                                                       f'EXP_MIN_CUTOFFS = {"A non-None OBJECT" if not GIVEN_MIN_CUTOFFS is None else None}\n',
                                                                       f'EXP_USE_OTHER = {"A non-None OBJECT" if not GIVEN_USE_OTHER is None else None}\n',
                                                                       f'EXP_START_LAG = {"A non-None OBJECT" if not GIVEN_START_LAG is None else None}\n',
                                                                       f'EXP_END_LAG = {"A non-None OBJECT" if not GIVEN_END_LAG is None else None}\n',
                                                                       f'EXP_SCALING = {"A non-None OBJECT" if not GIVEN_SCALING is None else None}\n'
                                                                       )

                                                                    print(*expected_output)

                                                                    Compiler = cfso.CompileFullSupportObject(
                                                                                         FULL_SUPPORT_OBJECT=msod.build_empty_support_object(_columns),
                                                                                         HEADER=GIVEN_HEADER,
                                                                                         VALIDATED_DATATYPES=GIVEN_VALIDATED_DATATYPES,
                                                                                         MODIFIED_DATATYPES=GIVEN_MODIFIED_DATATYPES,
                                                                                         FILTERING=GIVEN_FILTERING,
                                                                                         MIN_CUTOFF=GIVEN_MIN_CUTOFFS,
                                                                                         USE_OTHER=GIVEN_USE_OTHER,
                                                                                         START_LAG=GIVEN_START_LAG,
                                                                                         END_LAG=GIVEN_END_LAG,
                                                                                         SCALING=GIVEN_SCALING
                                                                                         )

                                                                    GIVEN_FULL_SUPPORT_OBJECT = Compiler.SUPPORT_OBJECTS

                                                                    del Compiler

                                                                    Dummy = ExpandCategoriesMLPackage(
                                                                                 GIVEN_DATA_OBJECT,
                                                                                 data_given_orientation,
                                                                                 data_return_orientation,
                                                                                 data_return_format,
                                                                                 FULL_SUPPORT_OBJECT=GIVEN_FULL_SUPPORT_OBJECT,
                                                                                 CONTEXT=GIVEN_CONTEXT,
                                                                                 KEEP=GIVEN_KEEP,

                                                                                 TARGET=None,
                                                                                 target_given_orientation=None,
                                                                                 TARGET_TRANSPOSE=None,
                                                                                 target_transpose_given_orientation=None,
                                                                                 TARGET_AS_LIST=None,
                                                                                 target_as_list_given_orientation=None,
                                                                                 target_is_multiclass=None,

                                                                                 address_multicolinearity=auto_drop_rightmost_column,
                                                                                 multicolinearity_cycler=False,
                                                                                 auto_drop_rightmost_column=auto_drop_rightmost_column,
                                                                                 append_ones_for_cycler=False,  # BUILD EXPS PORTION OF TEST IS NOT SET UP TO HANDLE THIS

                                                                                 prompt_to_edit_given_mod_dtypes=False, #True,
                                                                                 print_notes=True,
                                                                                 bypass_validation=bypass_validation,
                                                                                 prompt_user_for_accept=False, #True,
                                                                                 calling_module=calling_module,
                                                                                 calling_fxn=calling_fxn
                                                                    )


                                                                    act_calling_module = Dummy.calling_module
                                                                    act_calling_fxn = Dummy.calling_fxn
                                                                    act_data_given_orientation = Dummy.data_given_orientation
                                                                    act_data_given_format = Dummy.data_given_format
                                                                    act_data_return_orientation = Dummy.data_return_orientation
                                                                    act_data_return_format = Dummy.data_return_format
                                                                    act_auto_drop_rightmost_column = Dummy.auto_drop_rightmost_column
                                                                    ACT_DATA_OBJECT = Dummy.DATA_OBJECT
                                                                    ACT_VALIDATED_DATATYPES = Dummy.VALIDATED_DATATYPES
                                                                    ACT_MODIFIED_DATATYPES = Dummy.MODIFIED_DATATYPES
                                                                    ACT_FILTERING = Dummy.FILTERING
                                                                    ACT_MIN_CUTOFFS = Dummy.MIN_CUTOFFS
                                                                    ACT_USE_OTHER = Dummy.USE_OTHER
                                                                    ACT_CONTEXT = Dummy.CONTEXT
                                                                    ACT_KEEP = Dummy.KEEP
                                                                    ACT_START_LAG = Dummy.START_LAG
                                                                    ACT_END_LAG = Dummy.END_LAG
                                                                    ACT_SCALING = Dummy.SCALING
                                                                    ACT_HEADER = Dummy.OBJECT_HEADER

                                                                    NAMES = [
                                                                            'calling_module',
                                                                            'calling_fxn',
                                                                            'data_given_orientation',
                                                                            'data_given_format',
                                                                            'data_return_orientation',
                                                                            'data_return_format',
                                                                            'auto_drop_rightmost_column',
                                                                            'DATA',
                                                                            'HEADER',
                                                                            'VALIDATED_DATATYPES',
                                                                            'MODIFIED_DATATYPES',
                                                                            'FILTERING',
                                                                            'MIN_CUTOFFS',
                                                                            'USE_OTHER',
                                                                            'CONTEXT',
                                                                            'KEEP',
                                                                            'START_LAG',
                                                                            'END_LAG',
                                                                            'SCALING'
                                                                    ]

                                                                    EXPECTED_OUTPUTS = [
                                                                            exp_calling_module,
                                                                            exp_calling_fxn,
                                                                            exp_data_given_orientation,
                                                                            exp_data_given_format,
                                                                            exp_data_return_orientation,
                                                                            exp_data_return_format,
                                                                            exp_auto_drop_rightmost_column,
                                                                            EXP_DATA_OBJECT,
                                                                            EXP_HEADER,
                                                                            EXP_VALIDATED_DATATYPES,
                                                                            EXP_MODIFIED_DATATYPES,
                                                                            EXP_FILTERING,
                                                                            EXP_MIN_CUTOFFS,
                                                                            EXP_USE_OTHER,
                                                                            EXP_CONTEXT,
                                                                            EXP_KEEP,
                                                                            EXP_START_LAG,
                                                                            EXP_END_LAG,
                                                                            EXP_SCALING
                                                                    ]

                                                                    ACTUAL_OUTPUTS = [
                                                                            act_calling_module,
                                                                            act_calling_fxn,
                                                                            act_data_given_orientation,
                                                                            act_data_given_format,
                                                                            act_data_return_orientation,
                                                                            act_data_return_format,
                                                                            act_auto_drop_rightmost_column,
                                                                            ACT_DATA_OBJECT,
                                                                            ACT_HEADER,
                                                                            ACT_VALIDATED_DATATYPES,
                                                                            ACT_MODIFIED_DATATYPES,
                                                                            ACT_FILTERING,
                                                                            ACT_MIN_CUTOFFS,
                                                                            ACT_USE_OTHER,
                                                                            ACT_CONTEXT,
                                                                            ACT_KEEP,
                                                                            ACT_START_LAG,
                                                                            ACT_END_LAG,
                                                                            ACT_SCALING
                                                                    ]

                                                                    for description, expected_thing, actual_thing in zip(NAMES, EXPECTED_OUTPUTS, ACTUAL_OUTPUTS):

                                                                        try:
                                                                            is_equal = np.array_equiv(expected_thing, actual_thing)
                                                                            # print(f'\033[91m\n*** TEST EXCEPTED ON np.array_equiv METHOD ***\033[0m\x1B[0m\n')
                                                                        except:
                                                                            # try:
                                                                            is_equal = expected_thing == actual_thing
                                                                            # except:
                                                                            #     print(f'{description}:')
                                                                            #     print(f'\n\033[91mEXP_VALUE = \n{expected_thing}\033[0m\x1B[0m\n')
                                                                            #     print(f'\n\033[91mACT_VALUE = \n{actual_thing}\033[0m\x1B[0m\n')
                                                                            #     wls.winlinsound(888, 500)
                                                                            #     raise Exception(f'\n*** TEST FAILED "==" METHOD ***\n')

                                                                        if not is_equal:
                                                                            # print(f'*' * 90)
                                                                            print(f'Failed on trial {ctr} of at most {total_trials:,}')
                                                                            print(*expected_output)
                                                                            print()
                                                                            # print(f'\n\033[91mEXP_VALUE = \n{expected_thing}\033[0m\x1B[0m\n')
                                                                            # print(f'\n\033[91mACT_VALUE = \n{actual_thing}\033[0m\x1B[0m\n')
                                                                            print(f'\n\033[91mGIVEN_DATA_OBJECT = \n{GIVEN_DATA_OBJECT}\033[0m\x1B[0m\n')
                                                                            print(f'\n\033[91mEXP_DATA_OBJECT = \n{EXP_DATA_OBJECT}\033[0m\x1B[0m\n')
                                                                            print(f'\n\033[91mACT_DATA_OBJECT = \n{ACT_DATA_OBJECT}\033[0m\x1B[0m\n')
                                                                            time.sleep(1)
                                                                            wls.winlinsound(888, 500)
                                                                            raise Exception(
                                                                                f'\n*** {description} FAILED EQUALITY TEST, \nexpected = \n{expected_thing}\n'
                                                                                f'actual = \n{actual_thing} ***\n')
                                                                        else:
                                                                            pass  # print(f'\033[92m     *** {description} PASSED ***\033[0m\x1B[0m')

    print(f'\n\033[92m*** VALIDATION COMPLETED SUCCESSFULLY. ALL PASSED. ***\033[0m\x1B[0m\n')
    for _ in range(3): wls.winlinsound(888, 500); time.sleep(.5)











