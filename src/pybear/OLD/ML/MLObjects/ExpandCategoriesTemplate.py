import numpy as np, sparse_dict as sd, os, sys, inspect, time
from copy import deepcopy
if os.name == 'nt': import tensorflow_hub as hub   # BEAR, IF EVER RESOLVE LINUX tf ISSUES
import MemSizes as ms
from general_text import TextCleaner as tc
from general_list_ops import list_select as ls
from debug import get_module_name as gmn
from data_validation import validate_user_input as vui, arg_kwarg_validater as akv
from ML_PACKAGE._data_validation import list_dict_validater as ldv
from general_data_ops import get_dummies as gd
from MLObjects.SupportObjects import master_support_object_dict as msod, ModifiedDatatypes as md, FullSupportObjectSplitter as fsos
from MLObjects.ObjectOrienter import MLObjectOrienter as mloo
from MLObjects.PrintMLObject import print_object_preview as prop


#   THIS IS A PARENT CLASS TO USE IN ML PreRunExpandCategories AND FOR EXPANDING SRNL IN general_test.CreateSRNL. STOP.
#   HAS TO ACCOMMODATE BOTH, PreRun DOESNT HAVE SCALING & LAG, BUT CreateSRNL DOES. STOP.











##########################################################################################################################
##########################################################################################################################
# LAYOUT #################################################################################################################

# 12/26/22 BEAR UPDATE

# WITHIN __init__
# ******* init-type OPERATIONS ******************************************************************************************
#       I)   COMPULSORY OBJECT VALIDATION / PRELIMINARY (KW)ARG VALIDATION
#       II)  COMPULSORY STR-TYPE EXPANSION MENU DECLARATIONS
#       III) OPTIONAL MASS KWARG VALIDATION
#       IV)  DATA/TARGET OBJECT ORIENTATION - MLObjectOrienter
#            A) COMPULSORY OBJECT ORIENTATION FOR WIP PROCESSING OF DATA, FINALIZE FORMAT/ORIENTATION OF TARGET FOR MLReg
#            B) OPTIONAL TARGET DIMENSION VALIDATION
#       VI)  BUILD MODIFIED_DTYPES FROM GIVEN OR FROM DATA WITH OPTIONAL VALIDATION (ALL HANDLED BY ModifiedDatatypes)
#       VIII)DECLARATIONS / GIVEN OBJECT AND (KW)ARG BACKUP FOR RESET
#       IX)  BUILD HOLDER OBJECTS USED FOR AVOIDING REDUNDANT CALCULATIONS
#            A) BUILD UNIQUES (LIST OF UNIQUES IN EACH COLUMN -- LEN==#COLUMNS, IS ALL EMPTY EXCEPT THE SLOTS FOR STR-TYPE COLUMNS)
#            B) WORD_COUNT (TOTAL UNIQUES IN EACH EXAMPLE -- LEN==#COLUMNS, IS ALL EMPTY EXCEPT FOR STR-TYPE COLUMNS)
#            C) SPLIT HOLDER OBJECTS (mxn = #ROWS x #SPLIT_STR COLUMNS)
# ******* END init-type OPERATIONS ******************************************************************************************
# ******* Expander Preparations ************************************************************************************************
#       X)   IF TYPE IS "STR" OR "SPLIT_STR" GET UNIQUES
#       XI)  XI - LOGIC FOR WHETHER OR NOT AND HOW TO ADDRESS MULTICOLINEARITY
#            A) FINALIZE self.address_multicolinearity
#            B) LOGIC FOR auto_drop, cycler, AND append_ones_for_cycler
#       XII)  DETERMINE data_return_format, WHETHER OR NOT TO GENERATE MEMORY ESTIMATES
#       XIII) CREATE EMPTY OBJECT TO CATCH EXPANDED COLUMNS (MUST DO THIS BECAUSE CATCHER COULD BE A SPARSE_DICT)
# ******* END Expander Preparations ************************************************************************************************
# ******* Expander ***********************************************************************************************************
#       XIV) EXPANDER
#       Loop  (ONLY ONE CAN BE ACTUATED PER PASS BASED ON WHAT THE COLUMN DTYPE IS IN MODIFIED_DATATYPES)
#            A) INT,BIN,FLOAT HANDLING
#            B) NORMAL CATEGORICAL HANDLING (get_dummies)
#               1) RUN core_expansion
#               2) MULTICOLINEARITY HANDLING
#                  a) AUTODROP RIGHTMOST - auto_drop_rightmost_column, self.context_update_for_column_drop
#                  b) CYCLER - self.whole_data_object_stats(), self.column_drop_iterator(), self.whole_data_object_stats(),
#                              self.context_update_for_column_drop()
#            C) SPLIT_STR EXPAND (whatever)
#            D) NNLM HANDLING (nnlm50)
#            E) PLACEHOLDER FOR FUTURE TXT ANALYTICS HANDLING
#
#            F) UPDATE HEADER
#                 1) INT, BIN, FLOAT - NO UPDATE
#                 2) STR - DELETE THE UNEXPANDED COLUMN NAME FROM HEADER THEN INSERT THE EXPANDED HEADER NAMES FROM get_dummies
#                 3) SPLIT_STR - DELETE THE UNEXPANDED COLUMN NAME FROM HEADER, INSERT EXPANDED HEADER FROM whatever
#                 4) nnlm50 - DELETE THE UNEXPANDED COLUMN NAME FROM HEADER, INSERT EXPANDED HEADER FROM nnlm50
#            H) UPDATE MODIFIED_DATATYPES
#                 1) INT, BIN, FLOAT - NO UPDATE
#                 2) STR - INSERT 'BIN' FOR EACH COLUMN IN "LEVELS"
#                 3) SPLIT_STR - INSERT 'BIN' FOR EACH COLUMN IN "LEVELS"
#                 4) nnlm50 - INSERT 50 'FLOAT' COLUMNS
#            I) UPDATE VALIDATED_DATATYPES
#                 INT, BIN, FLOAT, STR, SPLIT_STR, nnlm50 - ALL JUST REPLICATE ORIG VALUE INTO NEW POSISTION
#            J) PUT LEVELS AT FRONT OF NEW_OBJECT_HOLDER
#                 1) INT, BIN, FLOAT - JUST PUT A COPY OF COLUMN IN NEW_OBJECT_HOLDER
#                 2) STR - UPDATE NEW_OBJECT_HOLDER W LEVELS FROM get_dummies
#                 3) SPLIT_STR - UPDATE NEW_OBJECT_HOLDER W LEVELS FROM whatever
#                 4) nnlm50 - UPDATE NEW_OBJECT_HOLDER W LEVELS FROM nnlm50
#            K) EMPTY self.LEVELS FOR NEXT ROUND
#        XV)  OPTIONAL PRINT STASTISTICS OF FINAL EXPANDED OBJECT
#        XVI) OPTIONAL PROMPT - USER ACCEPT OR RESTART
#        XVII)ORIENT FINAL OBJECT
#

# FUNCTIONS
# _exception()                           '''Exception verbage for this module.'''
# calculate_estimated_memory()           '''Only used once. Separate from code body for clarity.'''
# column_drop_iterator()                  # TO BE BROUGHT IN FROM MLRegression
# whole_data_object_stats()              '''Display statistics derived from MLRegression for passed object.'''
# context_update_for_column_drop()        # OVERWROTE IN CHILD




class ExpandCategoriesTemplate():
    '''Processed as [] = columns. Expands categorical columns and associated support objects.
        Requires at least modified datatypes (to know what columns to ignore/expand.) To willy-nilly
        expand any old data object as simply categorical, use general_data_ops.get_dummies.'''

    def __init__(self,
                 DATA_OBJECT,
                 data_given_orientation='COLUMN',
                 data_return_orientation='COLUMN',
                 data_return_format='PROMPT',
                 DATA_OBJECT_HEADER=None,
                 TARGET=None,   # MUST HAVE A TARGET TO DO FULL cycler, OTHERWISE CAN ONLY GET determ!!!
                 target_given_orientation=None,
                 TARGET_TRANSPOSE=None,
                 target_transpose_given_orientation=None,
                 TARGET_AS_LIST=None,
                 target_as_list_given_orientation=None,
                 target_is_multiclass=None,
                 SUPPORT_OBJECT_AS_MDTYPES_OR_FULL_SUP_OBJ=None,  # 3/2/23 VERIFIED VIA TEST CAN BE PASSED AS EITHER SINGLE MOD_DTYPE OR FULL SUP_OBJS
                 address_multicolinearity='PROMPT',
                 auto_drop_rightmost_column=False,
                 multicolinearity_cycler=True,
                 append_ones_for_cycler=True,
                 prompt_to_edit_given_mod_dtypes=False,
                 print_notes=False,
                 bypass_validation=False,
                 prompt_user_for_accept=True,
                 calling_module=None,
                 calling_fxn=None):


        while True:  # TO ALLOW USER ABORT, IF EXPANSION IS TOO BIG, FOR EXAMPLE, OR TO RESTART

            # *** init-type OPERATIONS **********************************************************************************************

            ##########################################################################################################################
            # I)   COMPULSORY OBJECT VALIDATION / PRELIMINARY (KW)ARG VALIDATION #####################################################
            self.data_given_format, self.DATA_OBJECT = ldv.list_dict_validater(DATA_OBJECT, 'DATA_OBJECT')

            self.target_given_format, TARGET = ldv.list_dict_validater(TARGET, 'TARGET')
            self.target_tranpose_given_format, TARGET_TRANSPOSE = ldv.list_dict_validater(TARGET_TRANSPOSE, 'TARGET_TRANSPOSE')
            self.target_as_list_given_format, TARGET_AS_LIST = ldv.list_dict_validater(TARGET_AS_LIST, 'TARGET_AS_LIST')

            self.this_module = gmn.get_module_name(str(sys.modules[__name__]))
            self.calling_module = calling_module if not calling_module is None else self.this_module
            self.calling_fxn = calling_fxn if not calling_fxn is None else inspect.stack()[0][3]

            bypass_validation = akv.arg_kwarg_validater(bypass_validation, 'bypass_validation',
                                        [True, False, None], self.calling_module, self.calling_fxn, return_if_none=False)

            # END I) COMPULSORY OBJECT VALIDATION / PRELIMINARY (KW)ARG VALIDATION #####################################################
            ##########################################################################################################################
            ##########################################################################################################################


            # MOD DTYPES WILL CHANGE TO ALL NUMBER DTYPES AFTER EXPANSION (INT, BIN, FLOAT) CANNOT HAVE STR DTYPES
            # 'STR' IS CONVERTED TO LEVELS BY get_dummies
            # 'SPLIT_STR' IS SPLIT, COMPILED TO GET UNIQUES, THEN CONVERTED TO LEVELS BY SOMETHING OTHER THAN get_dummies
            # (get_dummies USES np.where ON A COLUMN OF SINGLE WORDS TO FIND MATCHES AND BUILD 1/0 VECTORS)
            # 'NNLM50' USES tf nnlm_en_50 TO CONVERT ONE COLUMN OF PHRASES INTO AN ARRAY OF 50 COLUMNS OF FLOATS


            ##########################################################################################################################
            # II) COMPULSORY STR-TYPE EXPANSION MENU DECLARATIONS ####################################################################

            self.ALLOWED_MOD_DTYPES_LIST = list(msod.mod_text_dtypes().values()) + list(msod.mod_num_dtypes().values())

            # END II) COMPULSORY STR-TYPE EXPANSION MENU DECLARATIONS ################################################################
            ##########################################################################################################################


            ######################################################################################################################
            ######################################################################################################################
            # III OPTIONAL VALIDATION PART 1 - MASS KWARG VALIDATION #############################################################

            if bypass_validation:
                self.data_given_orientation = data_given_orientation
                self.data_return_orientation = data_return_orientation
                self.data_return_format = data_return_format
                self.address_multicolinearity = address_multicolinearity
                self.multicolinearity_cycler = multicolinearity_cycler
                self.auto_drop_rightmost_column = auto_drop_rightmost_column
                self.append_ones_for_cycler = append_ones_for_cycler

            elif not bypass_validation:
                self.data_given_orientation = akv.arg_kwarg_validater(data_given_orientation, 'data_given_orientation',
                                  ['ROW', 'COLUMN', None], self.calling_module, self.calling_fxn, return_if_none='COLUMN')
                self.data_return_orientation = akv.arg_kwarg_validater(data_return_orientation, 'data_return_orientation',
                                  ['ROW', 'COLUMN', None], self.calling_module, self.calling_fxn, return_if_none='COLUMN')

                target_is_multiclass = akv.arg_kwarg_validater(target_is_multiclass, 'target_is_multiclass',
                                  [True, False, None], self.calling_module, self.calling_fxn, return_if_none=False)
                target_given_orientation = akv.arg_kwarg_validater(target_given_orientation, 'target_given_orientation',
                                  ['ROW', 'COLUMN', None], self.calling_module, self.calling_fxn)

                target_transpose_given_orientation = akv.arg_kwarg_validater(target_transpose_given_orientation, 'target_transpose_given_orientation',
                                  ['ROW', 'COLUMN', None], self.calling_module, self.calling_fxn)
                target_as_list_given_orientation = akv.arg_kwarg_validater(target_as_list_given_orientation, 'target_as_list_given_orientation',
                                  ['ROW', 'COLUMN', None], self.calling_module, self.calling_fxn)

                self.data_return_format = akv.arg_kwarg_validater(data_return_format, 'data_return_format',
                                  ['ARRAY', 'SPARSE_DICT', 'PROMPT', None], self.calling_module, self.calling_fxn, return_if_none='PROMPT')
                self.address_multicolinearity = akv.arg_kwarg_validater(address_multicolinearity, 'address_multicolinearity',
                                  [True, False, 'PROMPT', None], self.calling_module, self.calling_fxn, return_if_none=False)
                self.multicolinearity_cycler = akv.arg_kwarg_validater(multicolinearity_cycler, 'multicolinearity_cycler',
                                  [True, False, 'PROMPT', None], self.calling_module, self.calling_fxn, return_if_none=False)
                self.append_ones_for_cycler = akv.arg_kwarg_validater(append_ones_for_cycler, 'append_ones_for_cycler',
                                  [True, False, 'PROMPT', None], self.calling_module, self.calling_fxn, return_if_none=False)
                self.auto_drop_rightmost_column = akv.arg_kwarg_validater(auto_drop_rightmost_column, 'auto_drop_rightmost_column',
                                  [True, False, 'PROMPT', None], self.calling_module, self.calling_fxn, return_if_none=False)
                print_notes = akv.arg_kwarg_validater(print_notes, 'print_notes',
                                  [True, False], self.calling_module, self.calling_fxn, return_if_none=False)
                prompt_to_edit_given_mod_dtypes = akv.arg_kwarg_validater(prompt_to_edit_given_mod_dtypes, 'prompt_to_edit_given_mod_dtypes',
                                  [True, False, None], self.calling_module, self.calling_fxn, return_if_none=False)
                prompt_user_for_accept = akv.arg_kwarg_validater(prompt_user_for_accept, 'prompt_user_for_accept',
                                  [True, False, None], self.calling_module, self.calling_fxn, return_if_none=False)


                if self.address_multicolinearity is False:
                    self.auto_drop_rightmost_column = False
                    self.multicolinearity_cycler = False
                    self.append_ones_for_cycler = False
                elif self.address_multicolinearity is True:
                    except_text = lambda x: print(f'USER HAS OPTED FOR address_multicolinearity BUT BOTH auto_drop_rightmost_column AND '
                                                  f'multicolinearity_cycler ARE {x}. '
                                                  f'\nONLY ALLOWED CASES IF address_multicolinearity is True: '
                                                  f'\n1) ONE IS True AND ONE IS False \n2) ONE IS "PROMPT" AND THE OTHER CAN '
                                                  f'BE ANYTHING \n3) BOTH ARE "PROMPT"')
                    if (self.auto_drop_rightmost_column is False and self.multicolinearity_cycler is False): except_text('False')
                    elif (self.auto_drop_rightmost_column is True and self.multicolinearity_cycler is True): except_text('True')
                    del except_text

            # END III OPTIONAL VALIDATION PART 1 - MASS KWARG VALIDATION #############################################################
            ######################################################################################################################
            ######################################################################################################################


            ###################################################################################################################
            ###################################################################################################################
            # IV-A OBJECT ORIENTATION ###########################################################################################
            if print_notes: print(f'\nOrienting objects. Patience...')
            # ORIENT DATA, BUILD TARGETS
            # GET TARGET W/ [] or {}=ROW & TARGET_AS_LIST AS [] = ROW FOR MLReg, TARGET_TRANSPOSE DOESNT NEED TO BE RETURNED
            # OK TO MAKE TARGET IN ANY WAY WANTED HERE, ONLY USED FOR COLINEARITY TEST THEN DISCARDED (ORIGINAL "TARGET" JUST CARRIES THRU)

            a_single_class_target_object_is_provided = not target_is_multiclass and \
                                                       not (TARGET is None and TARGET_TRANSPOSE is None and TARGET_AS_LIST is None)

            _ObjectOrienter = mloo.MLObjectOrienter(
                # GIVEN DATA MUST BE PROCESSED AS NP ARRAY. SPARSE_DICT CANNOT HOLD STR VALUES SO IF IT DID COME IN AS SD,
                # THEN IT MUST BE ONLY NUM VALUES, CONVERT IT TO ARRAY HERE, AND ENSURE [] = COLUMN
                DATA=self.DATA_OBJECT,
                data_given_orientation=data_given_orientation,
                data_return_orientation='COLUMN',
                data_return_format='ARRAY',

                DATA_TRANSPOSE=None,
                data_transpose_given_orientation=None,
                data_transpose_return_orientation=None,
                data_transpose_return_format=None,

                XTX=None,
                xtx_return_format=None,

                XTX_INV=None,
                xtx_inv_return_format=None,

                target_is_multiclass=target_is_multiclass,
                TARGET=TARGET if not target_is_multiclass else None,
                target_given_orientation=target_given_orientation if not target_is_multiclass else None,
                target_return_orientation='ROW' if a_single_class_target_object_is_provided else None,
                # IF USER HARD PICKED DATA RETURN FORMAT ALREADY, SET TARGET TO THAT FORMAT, ELSE IF PROMPT FOR DATA RETURN FORMAT LATER, SET TO ARRAY FOR NOW THEN CHANGE LATER IF NEEDED
                target_return_format=None if not a_single_class_target_object_is_provided else self.data_return_format if not self.data_return_format=='PROMPT' else 'ARRAY',

                TARGET_TRANSPOSE=TARGET_TRANSPOSE if not target_is_multiclass else None,
                target_transpose_given_orientation=target_transpose_given_orientation if not target_is_multiclass else None,
                target_transpose_return_orientation=None,
                target_transpose_return_format=None,

                TARGET_AS_LIST=TARGET_AS_LIST if not target_is_multiclass else None,
                target_as_list_given_orientation=target_as_list_given_orientation if not target_is_multiclass else None,
                target_as_list_return_orientation='ROW' if a_single_class_target_object_is_provided else None,

                RETURN_OBJECTS=['DATA', 'TARGET', 'TARGET_AS_LIST'] if not TARGET is None else ['DATA'],

                bypass_validation=True,
                calling_module=self.this_module,
                calling_fxn=inspect.stack()[0][3]
            )

            del a_single_class_target_object_is_provided

            self.DATA_OBJECT = _ObjectOrienter.DATA
            self.TARGET = _ObjectOrienter.TARGET
            self.TARGET_AS_LIST = _ObjectOrienter.TARGET_AS_LIST

            del _ObjectOrienter
            if print_notes: print(f'Done orienting objects.')
            # END IV-A OBJECT ORIENTATION ###########################################################################################
            ###################################################################################################################
            ###################################################################################################################


            ######################################################################################################################
            ######################################################################################################################
            # IV-B OPTIONAL VALIDATION PART 2 - TARGET LEN VALIDATION ###################################################################

            if not bypass_validation:
                # IF self.TARGET AND self.TARGET_AS_LIST EXIST, THEY MUST BE []=ROW
                if not self.TARGET is None and len(self.TARGET) != len(self.DATA_OBJECT[0]):
                    self._exception(f'TARGET # EXAMPLES MUST BE EQUAL TO OBJECT # EXAMPLES')
                if not self.TARGET_AS_LIST is None and len(self.TARGET_AS_LIST) != len(self.DATA_OBJECT[0]):
                    self._exception(f'TARGET_AS_LIST # EXAMPLES MUST BE EQUAL TO OBJECT # EXAMPLES')

            # END IV-B OPTIONAL VALIDATION PART 2 - TARGET LEN VALIDATION ###############################################################
            ######################################################################################################################
            ######################################################################################################################

            ######################################################################################################################
            ######################################################################################################################
            # VI MAKE MODIFIED DATATYPES FROM GIVEN OR FROM DATA WITH OPTIONAL VALIDATION ########################################

            SupObjsClass = md.ModifiedDatatypes(
                                       OBJECT=self.DATA_OBJECT,
                                       object_given_orientation='COLUMN',
                                       columns=len(self.DATA_OBJECT),
                                       OBJECT_HEADER=DATA_OBJECT_HEADER,
                                       SUPPORT_OBJECT=SUPPORT_OBJECT_AS_MDTYPES_OR_FULL_SUP_OBJ,
                                       prompt_to_override=prompt_to_edit_given_mod_dtypes,
                                       return_support_object_as_full_array=True,
                                       bypass_validation=bypass_validation,
                                       calling_module=self.this_module,
                                       calling_fxn=inspect.stack()[0][3]
                                       )

            self.SUPPORT_OBJECTS = SupObjsClass.SUPPORT_OBJECT
            hdr_idx = msod.master_support_object_dict()["HEADER"]["position"]
            mdtypes_idx = msod.master_support_object_dict()["MODIFIEDDATATYPES"]["position"]

            del SupObjsClass

            if 'NNLM50' in self.SUPPORT_OBJECTS[mdtypes_idx]:
                embed = hub.load("https://tfhub.dev/google/nnlm-en-dim50/2")
            # END VI BUILD MODIFIED_DTYPES FROM GIVEN OR FROM DATA WITH OPTIONAL VALIDATION ###############################
            ###############################################################################################################
            ################################################################################################################


            ######################################################################################################################
            # VIII - DECLARATIONS / PLACEHOLDERS / BACKUPS ##############################################################################
            # THESE KWARGS ARE DYNAMIC, MAKE A COPY HERE IN CASE USER DOES A RESTART

            self.l_mem = 0
            self.sd_mem = 0

            self.CONTEXT_HOLDER = []   # CATCHES WHAT WOULD BE UPDATES TO "CONTEXT" IN MLPACKAGE; UPDATES CONTEXT AT THE END IF CONTEXT EXISTS, ELSE IS DELETED

            # PLACEHOLDER FOR RETURNS FROM get_dummies() ET AL (STRING HANDLERS)
            self.LEVELS = None
            EXPANDED_COLUMN_NAMES = None
            DROPPED_COLUMN_NAMES = None
            number_of_split_str = np.sum(np.int8(self.SUPPORT_OBJECTS[mdtypes_idx]=='SPLIT_STR'))
            self.dum_indicator = ' - '
            SUPPORT_OBJECTS_BACKUP = self.SUPPORT_OBJECTS.copy()

            # END VIII - DECLARATIONS / PLACEHOLDERS / BACKUPS ##############################################################################
            #####################################################################################################################

            # ******* END init-type OPERATIONS ********************************************************************************************



            # ******* Expander Preparations ************************************************************************************************

            #######################################################################################################################
            # IX)  BUILD HOLDER OBJECTS USED FOR AVOIDING REDUNDANT CALCULATIONS ##################################################
            self.UNIQUES_HOLDER = np.empty(len(self.DATA_OBJECT), dtype=object)  # MUST BE object, PROBABLY WILL BE RAGGED
            self.WORD_COUNT_HOLDER = np.zeros(len(self.DATA_OBJECT), dtype=np.int16)  # USED ONLY FOR SPLIT_STRING IN ESTIMATING SIZE OF SD
            self.SPLIT_HOLDER = np.empty((number_of_split_str, len(self.DATA_OBJECT[0])), dtype=object)
                # HOLDS SPLIT WORDS FOR ALL SPLIT_STR COLUMNS, OTHERWISE WOULD HAVE TO DO THIS TWICE, HERE PRE-MEM ESTIMATE THEN LATER
                # WHEN GETTING COUNTS AND PUTTING IN self.LEVELS
            # IX) END BUILD HOLDER OBJECTS USED FOR AVOIDING REDUNDANT CALCULATIONS ##################################################
            #######################################################################################################################

            #######################################################################################################################
            #######################################################################################################################
            # X - IF TYPE IS "STR" OR "SPLIT_STR" GET UNIQUES  (THIS ALWAYS MUST BE DONE) #########################################
            # KEEP GENERATED LISTS OF UNIQUES FOR FAST BUILD-OUT OF self.LEVELS ###################################################
            # NEED TO DO THIS FIRST BECAUSE calculate_estimated_memory NEEDS self.UNIQUES_HOLDER & self.WORD_COUNT_HOLDER #########
            if print_notes: print(f'\n' + '*' * 90); print(f'Generating uniques for every STR-type column...')

            split_str_col_idx = -1
            for col_idx in range(len(self.DATA_OBJECT)-1, -1, -1):    # DATA MUST BE []=COLUMN BY THIS POINT
                # WHEN DOING THE EXPANSION, MUST GO THRU OBJECT BACKWARDS SO THAT SUPPORT OBJECTS CAN BE EXPANDED
                # WITHOUT DISRUPTION TO THE ORDINALITY OF col_idx ITERATOR DURING EXPANSION. THIS PROCESS OF FINDING
                # STRS AND GETTING UNIQUES MUST BE ITERATED IN THE SAME ORDER AS THE EXPANSION. SPLIT_HOLDER IS ALWAYS
                # FILLED FROM LEFT TO RIGHT REGARDLESS OF WHAT ORDER DATA IS ITERATED BECAUSE ITS ON A SEPARATE COUNTER.
                # SO THE DIRECTIONALITY THAT IT IS READ UNDER MUST BE THE SAME AS ITS FILLED UNDER.

                if self.SUPPORT_OBJECTS[mdtypes_idx][col_idx] in msod.mod_num_dtypes():
                    self.UNIQUES_HOLDER[col_idx] = np.empty((1, 0), dtype=np.int8)[0]

                elif self.SUPPORT_OBJECTS[mdtypes_idx][col_idx] == 'NNLM50':
                    # DONT USE TextCleaner HERE DATA SHOULD ALREADY BE CLEANED
                    # DONT OVERWRITE WHAT IS IN THE COLUMN IN DATA_OBJECT
                    # MUST BE AS ONE STR FOR nnlm50
                    self.UNIQUES_HOLDER[col_idx] = np.empty((1, 0), dtype=np.int8)[0]

                elif self.SUPPORT_OBJECTS[mdtypes_idx][col_idx] == 'STR':
                    self.UNIQUES_HOLDER[col_idx] = np.unique(self.DATA_OBJECT[col_idx]).astype('<U10000')

                elif self.SUPPORT_OBJECTS[mdtypes_idx][col_idx] == 'SPLIT_STR':
                    split_str_col_idx += 1

                    # JUST USE THIS TO GET UNIQUES, DATA SHOULD BE CLEAN BY THIS POINT, THIS IS NOT A CLEANING MODULE
                    SplitStrCleaner = tc.TextCleaner(self.DATA_OBJECT[col_idx], update_lexicon=False)
                    # DO THIS JUST FOR FUN, IF ALREADY CLEANED THIS WONT CHANGE ANYTHING
                    SplitStrCleaner.remove_characters()
                    SplitStrCleaner._strip()
                    SplitStrCleaner.normalize()

                    # DONT OVERWRITE THE EXISTING COLUMN IN DATA

                    SplitStrCleaner.as_list_of_lists()
                    self.UNIQUES_HOLDER[col_idx] = SplitStrCleaner.return_overall_uniques(return_counts=False)
                    self.SPLIT_HOLDER[split_str_col_idx] = SplitStrCleaner.CLEANED_TEXT
                    self.WORD_COUNT_HOLDER[col_idx] = sum(map(len, SplitStrCleaner.return_row_uniques(return_counts=False)))

                    del SplitStrCleaner

                # PLACEHOLDERS FOR POTENTIAL FUTURE TEXT ANALYTICS METHODS
                # elif self.SUPPORT_OBJECTS[mdtypes_idx][col_idx] == 'TXT4': pass
                # elif self.SUPPORT_OBJECTS[mdtypes_idx][col_idx] == 'TXT5': pass

            del split_str_col_idx

            if print_notes: print(f'Done.'); print(f'*'*90)
            # END X - GET COLUMN TYPES AND UNIQUES IF TYPE IS "STR" OR "SPLIT_STR" (THIS ALWAYS MUST BE DONE) ##############
            ################################################################################################################
            ################################################################################################################

            ################################################################################################################
            # XI - LOGIC FOR WHETHER OR NOT AND HOW TO ADDRESS MULTICOLINEARITY ##############################################
            # THIS MUST BE BEFORE calculate_estimated_memory, WHICH NEEDS self.address_multicolinearity
            # TO BE A BOOLEAN AND NOT 'PROMPT' IN ORDER TO CALCULATE SIZE
            # XI A - FINALIZE self.address_multicolinearity #######################################################################
            if self.address_multicolinearity in [True, False]:  # IF USER ALREADY HARD SELECTED THIS, JUST DO WHAT THEY WANT
                pass
            elif self.address_multicolinearity == 'PROMPT':     # IF PROMPT, FORCE DECISION ON HOW TO ADDRESS
                # STUFF FOR MANAGING HOW TO ADDRESS MULTICOLIN
                self.address_multicolinearity = {'Y': True, 'N': False}[
                    vui.validate_user_str(f'\nADDRESS MULTICOLINEARITY? (y/n) > ', 'YN')]
                if not self.address_multicolinearity:
                    self.multicolinearity_cycler, self.append_ones_for_cycler, self.auto_drop_rightmost_column = False, False, False
            else: akv.arg_kwarg_validater(self.address_multicolinearity, 'address_multicolinearity',
                [True, False, 'PROMPT'], self.this_module, inspect.stack()[0][3])    # akv FOR CATCHING / REPORTING BAD ARG
            # END XI A - FINALIZE self.address_multicolinearity #######################################################################

            # XI B - LOGIC FOR auto_drop, cycler, AND append_ones_for_cycler ##################################################
            # IF USER OPTED TO ADDRESS COLIN, DO LOGIC FOR auto_drop, cycler, AND append_ones_for_cycler ##################
            if self.address_multicolinearity:
                # VALIDATION AT TOP DOESNT ALLOW auto_drop AND cycler TO BE BOTH True OR BOTH False
                # IF USER ALREADY HARD SELECTED FOR OR AGAINST THIS, JUST DO WHAT THEY WANT
                if self.auto_drop_rightmost_column != 'PROMPT' and self.multicolinearity_cycler != 'PROMPT': pass
                else:  # auto_drop_rightmost_column == 'PROMPT' or multicolinearity_cycler == 'PROMPT':
                    while True:
                        __ = vui.validate_user_str(f'\nAuto-drop rightmost column(r), column drop cycler(c), or abort(a)? > ', 'RCA')
                        if __ == 'R':
                            self.auto_drop_rightmost_column = True
                            self.multicolinearity_cycler = False
                            self.append_ones_for_cycler = False
                            break
                        elif __ == 'C':
                            self.auto_drop_rightmost_column = False
                            self.multicolinearity_cycler = True
                            if self.append_ones_for_cycler in [True, False]: pass
                            else: self.append_ones_for_cycler = {'Y':True,'N':False}[
                                                        vui.validate_user_str(f'\nAppend ones for cycler (y/n) > ', 'YN')]
                            break
                        elif __ == 'A':
                            if vui.validate_user_str(f'Abort will disable all multicolinearity handling. Proceed? (y/n) > ', 'YN') == 'Y':
                                self.address_multicolinearity = False
                                self.multicolinearity_cycler = False
                                self.auto_drop_rightmost_column = False
                                self.append_ones_for_cycler = False
                                break
                            else: continue
            # END XI B - LOGIC FOR auto_drop, cycler, AND append_ones_for_cycler ################################################

            # VERIFY ALL MULTICOLIN ARGS ARE True/False AFTER ABOVE HANDLING #################################################
            akv.arg_kwarg_validater(self.address_multicolinearity, 'address_multicolinearity', [True, False],
                                    self.this_module, inspect.stack()[0][3])
            akv.arg_kwarg_validater(self.auto_drop_rightmost_column, 'auto_drop_rightmost_column', [True, False],
                                    self.this_module, inspect.stack()[0][3])
            akv.arg_kwarg_validater(self.multicolinearity_cycler, 'multicolinearity_cycler', [True, False],
                                    self.this_module, inspect.stack()[0][3])
            akv.arg_kwarg_validater(self.append_ones_for_cycler, 'append_ones_for_cycler', [True, False],
                                    self.this_module, inspect.stack()[0][3])
            # END VERIFY ALL MULTICOLIN ARGS ARE True/False AFTER ABOVE HANDLING #################################################

            # END XI - LOGIC FOR WHETHER OR NOT AND HOW TO ADDRESS MULTICOLINEARITY ##########################################
            ################################################################################################################

            # XII)  DETERMINE data_return_format, WHETHER OR NOT TO GENERATE MEMORY ESTIMATES ##############################
            if self.data_return_format in ['ARRAY', 'SPARSE_DICT']:  # IF USER ALREADY HARD SELECTED THIS, JUST DO WHAT THEY WANT
                pass
            elif self.data_return_format == 'PROMPT':     # IF PROMPT, GENERATE MEM USAGE FOR DECIDING WHAT FORMAT TO EXPAND AS

                self.calculate_estimated_memory()

                __ = vui.validate_user_str(f'Expand OBJECT as array(a) or sparse dict(s), restart(r), abort(t), quit(q) > ', 'ASRTQ')
                if __ == 'A': self.data_return_format = 'ARRAY'
                elif __ == 'S': self.data_return_format = 'SPARSE_DICT'
                elif __ == 'R': continue   # SEND BACK TO TOP OF WHILE LOOP
                elif __ == 'T': break # THIS SHOULD SEND OUT OF THIS MODULE WITHOUT ANY CHANGES EXCEPT THOSE MADE BY ObjectOrienter
                elif __ == 'Q': self._exception(f'USER TERMINATED.')


            else: akv.arg_kwarg_validater(self.data_return_format, 'data_return_format', ['ARRAY', 'SPARSE_DICT', 'PROMPT'],
                                  self.this_module, inspect.stack()[0][3])    # akv FOR CATCHING / REPORTING BAD ARG
            # END XII - DETERMINE self.data_return_format, WHETHER OR NOT TO GENERATE MEMORY ESTIMATES ##############################


            # XIII - CREATE EMPTY OBJECT TO CATCH EXPANDED COLUMNS ###########################################################
            # MUST USE A HOLDER OBJECT BECAUSE NEW OBJECT COULD BE SPARSE DICT; SUPPORT OBJS DO NOT NEED HOLDERS,
            # EDITS WILL BE MADE TO THE ORIGINAL OBJECTS

            if self.data_return_format == 'ARRAY':  # IF NOT RETURNING A SPARSE DICT, INSTANTIATE AN EMPTY NUMPY
                __ = self.SUPPORT_OBJECTS[mdtypes_idx]
                if 'FLOAT' in __ or 'NNLM50' in __: _dtype = np.float64
                elif 'INT' in __ or 'SPLIT_STR' in __: _dtype = np.int32
                elif 'STR' in __ or 'BIN' in __: _dtype = np.int8
                NEW_OBJECT_HOLDER = np.empty((0, len(self.DATA_OBJECT[0])), dtype=_dtype)
                del _dtype

            elif self.data_return_format == 'SPARSE_DICT':  # IF RETURNING A SPARSE DICT, INSTANTIATE AN EMPTY SD
                NEW_OBJECT_HOLDER = {}
            else: self._exception(f'INVALID VALUE "{self.data_return_format}" FOR data_return_format')
            # END XIII - CREATE EMPTY OBJECT TO CATCH EXPANDED COLUMNS #######################################################

            # ******* END Expander Preparations **************************************************************************************


            ###################################################################################################################
            ###################################################################################################################
            # XIV - EXPANDER #################################################################################################
            number_of_split_str_processed = -1
            for col_idx in range(len(self.DATA_OBJECT)-1, -1, -1):
                # AT THIS POINT, DATA_OBJECT_WIP MUST BE A NUMPY OF RAW DATA WITH [] = COLUMN.
                # ITERATE BACKWARDS SO THAT SUPPORT OBJECTS CAN BE EXPANDED INSITU W/O IMPACTING col_idx.
                # PROCESS EACH COLUMN BASED ON MOD_DTYPE
                # --- AUTOMATICALLY PUT FLOAT & INT COLUMNS INTO "DATA_HOLDER"
                # --- USE get_dummies TO EXPAND 'STR'
                # --- USE "WHATEVER" TO EXPAND 'SPLIT_STR'
                # --- nnlm50 TO EXPAND 'nnlm50'

                if print_notes: print(f"\nPROCESSING {self.SUPPORT_OBJECTS[hdr_idx][col_idx]}...")

                ###########################################################################################################
                # XIV A - NUMBER HANDLING #####################################################################################
                if self.SUPPORT_OBJECTS[mdtypes_idx][col_idx] in ['FLOAT', 'INT', 'BIN']:  # IF NUMBER, THEN JUST KEEP AS IS ##########
                    if print_notes: print(f'{self.SUPPORT_OBJECTS[hdr_idx][col_idx]} IS NOT CATEGORICAL')

                    if self.data_return_format == 'ARRAY':
                        self.LEVELS = self.DATA_OBJECT[col_idx].copy().reshape((1,-1))
                    elif self.data_return_format == 'SPARSE_DICT':
                        if self.SUPPORT_OBJECTS[mdtypes_idx][col_idx] == 'FLOAT':
                            self.LEVELS = sd.zip_list_as_py_float(self.DATA_OBJECT[col_idx].copy().reshape((1,-1)))
                        elif self.SUPPORT_OBJECTS[mdtypes_idx][col_idx] in ['BIN', 'INT']:
                            self.LEVELS = sd.zip_list_as_py_int(self.DATA_OBJECT[col_idx].copy().reshape((1,-1)))

                    # NO UPDATE TO HEADER AND OTHER SUPPORT OBJECTS, CREATE EMPTIES TO INSERT
                    # CREATE A COPY OF THE ORIGINAL TO RE-INSERT AT THE BOTTOM OF THIS
                    EXPANDED_COLUMN_NAMES = np.array(self.SUPPORT_OBJECTS[hdr_idx][col_idx], dtype='<U200').reshape((1,-1))
                    mdtype_update = self.SUPPORT_OBJECTS[mdtypes_idx][col_idx]
                # END XIV A - NUMBER HANDLING ################################################################################
                ###########################################################################################################


                ###########################################################################################################
                # XIV B - NORMAL CATEGORICAL HANDLING ###############################################################################
                elif self.SUPPORT_OBJECTS[mdtypes_idx][col_idx] == 'STR':  # IF COL IS CATEGORICAL, LET THE LEVEL FUN BEGIN
                    if print_notes:
                        print(f'CONVERTING CATEGORICAL COLUMN "{self.SUPPORT_OBJECTS[hdr_idx][col_idx]}" INTO COLUMNS OF DUMMIES...')

                    # GENERATE A DUMMY ARRAY BROKEN INTO LEVELS FOR THE COLUMN USING general_data_ops.get_dummies
                    # UPDATES self.LEVELS, EXPANDED_COLUMN_NAMES, DROPPED_COLUMN_NAMES
                    # THE selfs THAT MANAGE THE LOGIC FOR MULTICOLIN ARE ALSO FED AS KWARGS INTO THIS, SO THE OBJECT IS
                    # GUARANTEED TO BE FORMATTED CORRECTLY FOR WHATEVER OPERATION THE LOGIC ALLOWS IT TO DIVE INTO

                    ################################################################################################################
                    # XIV B 1 - CORE EXPANSION #######################################################################################

                    self.LEVELS, EXPANDED_COLUMN_NAMES, DROPPED_COLUMN_NAMES = \
                        gd.get_dummies(self.DATA_OBJECT[col_idx].reshape((1,-1)),
                                       OBJECT_HEADER=np.array(self.SUPPORT_OBJECTS[hdr_idx][col_idx]).reshape((1, -1)),
                                       given_orientation='COLUMN',
                                       IDXS=[0],
                                       UNIQUES=self.UNIQUES_HOLDER[col_idx].reshape((1,-1)),
                                       return_orientation='COLUMN',
                                       expand_as_sparse_dict=True if self.data_return_format == 'SPARSE_DICT' else False,
                                       auto_drop_rightmost_column=self.auto_drop_rightmost_column,
                                       append_ones=self.append_ones_for_cycler,
                                       bypass_validation=bypass_validation,
                                       bypass_sum_check=True,
                                       calling_module=self.this_module,
                                       calling_fxn=str(inspect.stack()[0][3])
                                       )

                    # END XIV B 1 - CORE EXPANSION #######################################################################################
                    ################################################################################################################

                    ################################################################################################################
                    # XIV B 2 - MULTICOLINEARITY HANDLING #######################################################################################
                    if not self.address_multicolinearity: pass

                    elif self.address_multicolinearity:

                        ########################################################################################################
                        # XIV B 2 a - HANDLING FOR AUTODROP ################################################################################
                        if self.auto_drop_rightmost_column:
                            # IF auto_drop IS True, get_dummies() WILL AUTOMATICALLY RETURN LEVELS W THE COLUMN DROPPED.
                            # HOLD UPDATES TO NEW_OBJECT_HOLDER UNTIL AFTER THE MULTICOLIN if/elif CHAIN

                            self.context_update_for_column_drop(f'Deleted {DROPPED_COLUMN_NAMES[0]} for multicolinearity auto_drop_rightmost_column')

                        # END XIV B 2 a HANDLING FOR AUTODROP ############################################################################
                        ########################################################################################################

                        ########################################################################################################
                        # XIV B 2 b HANDLING FOR CYCLER ##################################################################################
                        elif multicolinearity_cycler:
                            # HOLD UPDATES TO NEW_OBJECT_HOLDER UNTIL AFTER THE MULTICOLIN if/elif CHAIN

                            # THIS FUNCTION OUTPUT WILL CHANGE IF TARGET IS/ISNT PRESENT
                            self.whole_data_object_stats()
                            # self.whole_data_object_stats(self.LEVELS, EXPANDED_COLUMN_NAMES,
                            #                               self.UNIQUES_HOLDER[col_idx], append_ones='Y')

                            # BEAR COLUMN DROP ITERATOR IS GOING TO NEED []=COLUMNS TO DROP FAST FOR SD, GOOD THING ITS
                            # ALREADY COMING OUT OF get_dummies AS COLUMN. CANT GET AROUND
                            # HAVING TO MAKE XTX AND XT OVER AND OVER. TARGET IS ALREADY TRANSPOSED TO []=ROWS ABOVE

                            # 12/1/22 DECIDED THAT column_drop_iterator WILL ALWAYS RUN WITH WHATEVER return_as_sparse_dict
                            # SAYS. KNOW THAT ARRAY WILL ALWAYS BE FASTER, BUT IF ALWAYS MAKE LEVELS EXTRACT AS ARRAY THEN
                            # GO INTO iterator, TOO RISKY THAT EXTRACTING TOO BIG TO NUMPY COULD CRASH. TOO MUCH CODE
                            # GYMNASTICS TO CONDITIONALLY MANAGE SIZE & SPEED. JUST DONT MAKE SPARSE IF DONT REALLY NEED IT.

                            # THIS FUNCTION OUTPUT WILL CHANGE IF TARGET IS/ISNT PRESENT
                            self.column_drop_iterator()
                            # self.column_drop_iterator(self.LEVELS, EXPANDED_COLUMNS_NAMES,
                            #                               self.UNIQUES_HOLDER[col_idx], append_ones='Y')

                            # CREATE A MENU FOR list_select FROM UNIQUES LIST W MODIFICATION THAT GIVES FREQ & A 'NONE' OPTION ########
                            # KEEP THIS OUT OF THE BELOW while LOOP
                            __ = self.UNIQUES_HOLDER[col_idx]
                            UNIQUES_MENU = [f"{__[_]} ({np.sum(np.int8(self.DATA_OBJECT[col_idx].astype('<U500')==(__[_])))})"
                                           for _ in range(len(__))] + ['NONE']
                            del __
                            # END FANCY MENU LIST ######################################################################################

                            while True:
                                # ALLOW USER TO CHOOSE WHICH COLUMN TO DROP AND DISPLAY RESULT ON THE FLY
                                print(f'COLUMN "{self.SUPPORT_OBJECTS[msod.QUICK_POSN_DICT()["HEADER"]][col_idx]}"')

                                drop_idx = ls.list_single_select(UNIQUES_MENU, 'Select column to drop', 'idx')[0]

                                if drop_idx == len(UNIQUES_MENU)-1:
                                    if vui.validate_user_str(f'\nUser chose to not drop a column. Accept? (y/n) > ', 'YN') == 'Y':
                                        break

                                # SHOW RESULTS OF DROPPING COLUMN
                                if self.data_return_format == 'ARRAY':
                                    LEVELS_WIP = np.delete(self.LEVELS.copy(), drop_idx, axis=0)
                                elif self.data_return_format == 'SPARSE_DICT':
                                    LEVELS_WIP = sd.delete_outer_key(deepcopy(self.LEVELS), [drop_idx])[0]

                                UNIQUES_WIP = np.delete(self.UNIQUES_HOLDER[col_idx].copy(), drop_idx, axis=0)


                                self.whole_data_object_stats()
                                # self.whole_data_object_stats(LEVELS_WIP, EXPANDED_COLUMN_NAMES, UNIQUES_WIP, append_ones='Y')

                                if vui.validate_user_str(f'\nAccept column selection? (y/n) > ', 'YN') == 'Y':
                                    del LEVELS_WIP, UNIQUES_WIP
                                    break

                            # IF APPENDED ONES FOR CYCLER, REMOVE
                            if self.append_ones_for_cycler:
                                _ = len(self.LEVELS)-1
                                EXPANDED_COLUMN_NAMES = np.delete(EXPANDED_COLUMN_NAMES, _, axis=1)
                                if self.data_return_format=='ARRAY':
                                    self.LEVELS = np.delete(self.LEVELS, _, axis=0)
                                elif self.data_return_format=='SPARSE_DICT':
                                    self.LEVELS = sd.delete_outer_key(self.LEVELS, [_])[0]
                                del _

                            if drop_idx == len(UNIQUES_MENU)-1:  # IF THIS TRUE, USER SELECTED NOT TO DROP A COLUMN ABOVE
                                # SKIP BY AND USE UNALTERED "UNIQUES" AND "LEVELS"
                                pass
                            else:
                                # BEAR THIS BEING CREATED FOR FUTURE IF CHANGING ROWS OF DROPPED HITS TO "-1"
                                RETAINED_POP = deepcopy(self.LEVELS[drop_idx]) if self.data_return_format == 'SPARSE_DICT' else \
                                    self.LEVELS[drop_idx].copy()

                                # UPDATE "CONTEXT" W DROPPED COLUMN
                                DROPPED_COLUMN_NAMES = np.array([EXPANDED_COLUMN_NAMES[0][drop_idx]])

                                self.context_update_for_column_drop(f'User selected to drop {DROPPED_COLUMN_NAMES[0]} during multicolinearity cycler.')

                                self.UNIQUES_HOLDER[col_idx] = np.delete(self.UNIQUES_HOLDER[col_idx], drop_idx, axis=0)

                                EXPANDED_COLUMN_NAMES = np.delete(EXPANDED_COLUMN_NAMES, drop_idx, axis=1)

                                if self.data_return_format == 'ARRAY':
                                    self.LEVELS = np.delete(self.LEVELS, drop_idx, axis=0)
                                elif self.data_return_format == 'SPARSE_DICT':
                                    self.LEVELS = sd.delete_outer_key(self.LEVELS, [drop_idx])[0]

                                del RETAINED_POP

                            del UNIQUES_MENU
                        # END XIV B 2 b - HANDLING FOR CYCLER ##################################################################################
                        ############################################################################################################


                    # self.LEVELS IS SET BY get_dummies
                    # EXPANDED_COLUMN_NAMES IS SET BY get_dummies, BUT A COLUMN CAN BE DROPPED IF RUNNING cycler
                    mdtype_update = 'BIN'

                    # END XIV B 2 - MULTICOLINEARITY HANDLING #####################################################################
                    ################################################################################################################

                # END XIV B - NORMAL CATEGORICAL HANDLING #########################################################################
                ####################################################################################################################

                #####################################################################################################################
                # XIV C - SPLIT_STR CATEGORICAL HANDLING ####################################################################################
                elif self.SUPPORT_OBJECTS[mdtypes_idx][col_idx] == 'SPLIT_STR':
                    ## NOT ANTICIPATING ANY NEED TO ADDRESS MULTICOLINEARITY
                    number_of_split_str_processed += 1   # STARTS AT -1 ABOVE THIS for LOOP

                    if print_notes:
                        print(f'CONVERTING SPLIT_STR COLUMN "{self.SUPPORT_OBJECTS[hdr_idx][col_idx]}" INTO COLUMNS OF DUMMIES...')

                    if self.data_return_format == 'ARRAY':
                        self.LEVELS = np.zeros((len(self.UNIQUES_HOLDER[col_idx]), len(self.DATA_OBJECT[col_idx])), dtype=np.int16)
                    elif self.data_return_format == 'SPARSE_DICT':
                        self.LEVELS = {_:{} for _ in range(len(self.UNIQUES_HOLDER[col_idx]))}

                    _inner_len = len(self.DATA_OBJECT[col_idx])
                    for example_idx in range(_inner_len):
                        for unique_idx, unique in enumerate(self.UNIQUES_HOLDER[col_idx]):
                            _occurrences = np.sum(np.int8(self.SPLIT_HOLDER[number_of_split_str_processed][example_idx]==unique))
                            if _occurrences != 0:
                                self.LEVELS[int(unique_idx)][int(example_idx)] = int(_occurrences)  # BOTH ARRAY & SPARSEDICT

                    if self.data_return_format=='SPARSE_DICT':
                        for outer_idx in self.LEVELS:   # OUTER KEYS WERE SET AS INTS AT CREATION ABOVE
                            if _inner_len-1 not in self.LEVELS[outer_idx]:
                                self.LEVELS[int(outer_idx)][int(_inner_len-1)] = 0

                    del _inner_len, _occurrences

                    EXPANDED_COLUMN_NAMES = \
                        np.fromiter((f'{self.SUPPORT_OBJECTS[hdr_idx][col_idx]}{self.dum_indicator}{self.UNIQUES_HOLDER[col_idx][_]}'
                                     for _ in range(len(self.LEVELS))), dtype='<U200').reshape((1,-1))

                    mdtype_update = 'INT'
                # END XIV C - SPLIT_STR CATEGORICAL HANDLING ##################################################################################
                ############################################################################################################

                #####################################################################################################################
                # XIV D - GOOGLE EMBEDDED TEXT HANDLING ############################################################################
                elif self.SUPPORT_OBJECTS[mdtypes_idx][col_idx] == 'NNLM50':
                    if print_notes:
                        print(f'CONVERTING NNLM50 COLUMN "{self.SUPPORT_OBJECTS[hdr_idx][col_idx]}" INTO NNLM50 FLOATS...')

                    self.LEVELS = np.array(embed(self.DATA_OBJECT[col_idx]), dtype=np.float64).transpose()
                    if self.data_return_format == 'SPARSE_DICT':
                        self.LEVELS = sd.zip_list_as_py_float(self.LEVELS)
                    EXPANDED_COLUMN_NAMES = np.fromiter((f'{self.SUPPORT_OBJECTS[hdr_idx][col_idx]}_NNLM50_{_}' for _ in range(1,51)), dtype='<U200').reshape((1,-1))
                    mdtype_update = 'FLOAT'
                # END XIV D - GOOGLE EMBEDDED TEXT HANDLING ##########################################################################
                #######################################################################################################################

                ###################################################################################################################
                # XIV E - PLACEHOLDER FOR FUTURE TXT ANALYTICS HANDLING ##########################################################
                # PLACEHOLDERS FOR POTENTIAL FUTURE TEXT ANALYTICS METHODS
                # elif self.SUPPORT_OBJECTS[mdtypes_idx][col_idx] == 'TXT4': pass
                # elif self.SUPPORT_OBJECTS[mdtypes_idx][col_idx] == 'TXT5': pass
                # END XIV E - PLACEHOLDER FOR FUTURE TXT ANALYTICS HANDLING ######################################################
                ###################################################################################################################

                # XIV F - UPDATE DATA_HOLDER W LEVELS, UPDATE SUPPORT_OBJECTS W LEVEL HEADER, & NEW DTYPES ########################

                ACTIVE_SUPPORT_COLUMN = self.SUPPORT_OBJECTS[:, col_idx].reshape((-1,1))
                ACTIVE_SUPPORT_COLUMN[mdtypes_idx][0] = mdtype_update
                # REPLICATE ACTIVE_SUPPORT_COLUMN len(self.LEVELS) TIMES
                # AS OF 1/29/23 ORIGINAL VAL_DTYPES IS REPLICATED, MEANING STRS THAT BECAME MOD_DTYPE BIN ARE STILL VAL_DTYPE STR,
                # BELIEVE THIS IS HOW ITS BEEN DONE ALL ALONG

                EXPANSION_INSERT = np.empty((len(ACTIVE_SUPPORT_COLUMN), 0), dtype=object)
                for _ in self.LEVELS:
                    EXPANSION_INSERT = np.hstack((EXPANSION_INSERT, ACTIVE_SUPPORT_COLUMN))
                del ACTIVE_SUPPORT_COLUMN, mdtype_update

                # APPLY EXPANDED_COLUMN_NAMES TO SUPPORT_OBJECTS IN HEADER POSITION
                EXPANSION_INSERT[msod.master_support_object_dict()["HEADER"]["position"]] = EXPANDED_COLUMN_NAMES[0]

                # OVERWRITE CURRENT COLUMN IN SUPPORT_OBJECTS WITH EXPANSION
                self.SUPPORT_OBJECTS = np.hstack((self.SUPPORT_OBJECTS[:, :col_idx],
                                                  EXPANSION_INSERT,
                                                  self.SUPPORT_OBJECTS[:, col_idx+1:]
                                                  ))

                ######################################################################################################################
                # XIV J - PUT LEVELS AT FRONT OF NEW_OBJECT_HOLDER ###################################################################
                if self.data_return_format == 'ARRAY':
                    NEW_OBJECT_HOLDER = np.vstack((self.LEVELS, NEW_OBJECT_HOLDER))
                elif self.data_return_format == 'SPARSE_DICT':
                    NEW_OBJECT_HOLDER = sd.core_merge_outer(self.LEVELS, NEW_OBJECT_HOLDER)[0]
                # XIV J - PUT LEVELS AT FRONT OF NEW_OBJECT_HOLDER ###################################################################
                ######################################################################################################################

                # XIV K
                self.LEVELS = None  # FREE UP SOME MEMORY TEMPORARILY

            del number_of_split_str_processed

            # END XIV - EXPANDER #################################################################################################
            ###################################################################################################################
            ###################################################################################################################


            ###################################################################################################################
            # XV - OPTIONAL PRINT STASTISTICS OF FINAL EXPANDED OBJECT #######################################################
            if print_notes:
                print(f'\nExpanding columns done.')

                _is_dict = isinstance(NEW_OBJECT_HOLDER, dict)
                print(f'\n *** EXPANDED DATA HAS {len(NEW_OBJECT_HOLDER)} COLUMNS AND '
                      f'{sd.inner_len_quick(NEW_OBJECT_HOLDER) if _is_dict else len(NEW_OBJECT_HOLDER[0])} '
                      f'ROWS (NOT INCLUDING HEADER) ***')

                if self.l_mem != 0:   # FOR THIS TO BE VALID, expand_as_sparse_dict WOULD HAVE TO HAVE STARTED AS "PROMPT", CAUSING
                    # calculate_estimated_memory() TO TRIGGER AND OVERWRITE self.l_mem AND self.sd_mem WITH NON-ZERO NUMBERS.
                    # IF NOT "PROMPT", self.l_mem AND self.sd_mem WOULD STAY IN __init__ STATE OF ZERO, AND THE SIZE TEST BYPASSED.
                    print(f"\nACTUAL DATA OBJECT IS {'SPARSE DICT' if _is_dict else 'ARRAY'}, " +
                          f"SIZE BY getsizeof IS {round(sys.getsizeof(NEW_OBJECT_HOLDER) /1024**2, 4)} MB")
                    print(f'COMPARE WITH {round(self.sd_mem if _is_dict else self.l_mem, 4)} MB PREDICTED PRE-EXPANSION')

                del _is_dict
            # END XV - OPTIONAL PRINT STATISTICS OF FINAL EXPANDED OBJECT #######################################################
            ###################################################################################################################

            ###################################################################################################################
            # XVI - OPTIONAL PROMPT FOR USER ACCEPT OR RESTART ###############################################################
            if prompt_user_for_accept:
                print(f'\nPREVIEW OF DATA:')
                prop.print_object_preview(NEW_OBJECT_HOLDER, 'DATA', 20, 10, 0, 0, orientation='column',
                    header=np.array(self.SUPPORT_OBJECTS[hdr_idx]).reshape((1, -1))[0][:10])

                if vui.validate_user_str(f'\nAccept data expansion(a) or restart(r) > ', 'AR') == 'R':

                    def kwarg_changer(kwarg, name, ALLOWED_DICT):
                        ALLOWED_DICT = ALLOWED_DICT | {'@':'ABORT THIS CHANGE'}
                        while True:
                            if vui.validate_user_str(f'\n{name} is currently set to {kwarg}, change? (y/n) > ', 'YN') == 'Y':
                                _ = vui.validate_user_str(f'Change to '
                                    f'{", ".join([f"{v}({k.lower()})" for k, v in ALLOWED_DICT.items()])} > ', "".join(list(ALLOWED_DICT.keys())))
                            else: break
                            if _ == '@': break
                            if vui.validate_user_str(f'User entered "{ALLOWED_DICT[_]}" for {name}, accept? (y/n) > ', 'YN') == 'Y':
                                kwarg = ALLOWED_DICT[_]
                                del _, ALLOWED_DICT
                                break

                        return kwarg

                    data_return_format = kwarg_changer(data_return_format, 'data_return_format', {'S':'SPARSE_DICT', 'A':'ARRAY', 'P':'PROMPT'})
                    self.address_multicolinearity = kwarg_changer(address_multicolinearity, 'address_multicolinearity', {'T':True, 'F':False, 'P':'PROMPT'})
                    if self.address_multicolinearity:
                        auto_drop_rightmost_column = kwarg_changer(auto_drop_rightmost_column, 'auto_drop_rightmost_column', {'T':True, 'F':False, 'P':'PROMPT'})
                        multicolinearity_cycler = kwarg_changer(multicolinearity_cycler, 'multicolinearity_cycler', {'T':True, 'F':False, 'P':'PROMPT'})
                        if multicolinearity_cycler:
                            append_ones_for_cycler = kwarg_changer(append_ones_for_cycler, 'append_ones_for_cycler', {'T':True, 'F':False})
                    prompt_to_edit_given_mod_dtypes = kwarg_changer(prompt_to_edit_given_mod_dtypes, 'prompt_to_edit_given_mod_dtypes', {'T':True, 'F':False})
                    print_notes = kwarg_changer(print_notes, 'print_notes', {'T':True, 'F':False})
                    bypass_validation = kwarg_changer(bypass_validation, 'bypass_validation', {'T':True, 'F':False})
                    prompt_user_for_accept = kwarg_changer(prompt_user_for_accept, 'prompt_user_for_accept', {'T':True, 'F':False})

                    del kwarg_changer

                    self.SUPPORT_OBJECTS = SUPPORT_OBJECTS_BACKUP.copy()

                    continue  # GO BACK UP TO TOP-LEVEL while
            # END XVI - OPTIONAL PROMPT FOR USER ACCEPT OR RESTART ###############################################################
            ###################################################################################################################


            # ELSE IF BYPASS user_accept OR USER ACCEPTED
            ###################################################################################################################
            # XVI - ORIENT FINAL OBJECT ########################################################################################
            if self.data_return_orientation == 'COLUMN': pass
            elif self.data_return_orientation == 'ROW':
                if self.data_return_format == 'ARRAY': NEW_OBJECT_HOLDER = NEW_OBJECT_HOLDER.transpose()
                elif self.data_return_format == 'SPARSE_DICT': NEW_OBJECT_HOLDER = sd.core_sparse_transpose(NEW_OBJECT_HOLDER)
            # END XVI - ORIENT FINAL OBJECT ###################################################################################
            ###################################################################################################################


            # DONT DO A SUPPLEMENTAL DELETE HERE, THE SUPPLEMENTAL OBJECTS NEED TO BE AVAILABLE AS ATTRS OF THE CLASS!
            self.DATA_OBJECT = NEW_OBJECT_HOLDER
            del NEW_OBJECT_HOLDER, DATA_OBJECT

            fsos.FullSupObjSplitter.__init__(self, self.SUPPORT_OBJECTS, bypass_validation=bypass_validation)

            break  # BREAK TOP LEVEL while TO END init

    #END INIT #############################################################################################################
    #######################################################################################################################
    #######################################################################################################################





    def _exception(self, text):
        '''Exception verbage for this module.'''
        raise Exception(f'\n***{self.this_module}() THRU {self.calling_module}.{self.calling_fxn}() >>> {text} ***\n')


    def calculate_estimated_memory(self):
        '''Only used once. Separate from code body for clarity.'''
        ######################################################################################################################
        ######################################################################################################################
        # MEM USAGE ESTIMATES FOR EXPANSION AS ARRAY OR SPARSEDICT ###########################################################

        print(f'\n' + '*'*90)
        print(f'GENERATING ESTIMATES OF DATA SIZE IN RAM AFTER EXPANSION...')

        # INSTEAD OF MAKING THESE idxs A self. IN init AND CARRYING AROUND THE self. A HUNDRED TIMES, JUST REDECLARE HERE
        hdr_idx = msod.master_support_object_dict()["HEADER"]["position"]
        mdtypes_idx = msod.master_support_object_dict()["MODIFIEDDATATYPES"]["position"]

        np_float = ms.MemSizes('np_float').mb()
        np_int = ms.MemSizes('np_int').mb()
        sd_float = ms.MemSizes('sd_float').mb()
        sd_int = ms.MemSizes('sd_int').mb()

        l_rows, sd_rows = len(self.DATA_OBJECT[0]), len(self.DATA_OBJECT[0])
        l_end_cols, sd_end_cols, l_elems, sd_elems = 0, 0, 0, 0     # self.l_mem, self.sd_mem ALREADY __init__ed
        adjusted_columns = 0
        float_in_final_expanded = 'FLOAT' in self.SUPPORT_OBJECTS[mdtypes_idx] or 'NNLM50' in self.SUPPORT_OBJECTS[mdtypes_idx]  # IF A FLOAT COLUMN IN FINAL, ALL COLUMNS WILL BE float64
        for col_idx in range(len(self.DATA_OBJECT)):
            col_type = self.SUPPORT_OBJECTS[mdtypes_idx][col_idx]
            if col_type in ['FLOAT', 'INT', 'BIN']:
                # IF NUMBER PRE-EXPANSION, IS (NOT NECESSARILY, BUT PROBABLY) A FULLY DENSE COLUMN IN ARRAY OR SPARSEDICT.
                # INCREMENT cols, elems, mem
                l_end_cols += 1
                sd_end_cols += 1
                l_elems += l_rows
                sd_elems += sd_rows
                self.l_mem += l_rows * (np_float if float_in_final_expanded else np_int)   # IF A FLOAT COLUMN IN FINAL, ALL COLUMNS WILL BE float64
                if col_type in ['FLOAT']: self.sd_mem += sd_rows * sd_float
                elif col_type in ['INT', 'BIN']: self.sd_mem += sd_rows * sd_int

            elif col_type == 'STR':  # THIS WILL BE EXPANDED TO BIN, BEST CASE IS np.int8 OR py_int
                # end_cols FORMULA IS ( NUMBER AFTER EXPANDED - int(IF MULTICOLIN ADDRESSED VIA AUTODROP OR USER SELECT AFTER CYCLE) )
                adjusted_columns = len(self.UNIQUES_HOLDER[col_idx]) - int(self.address_multicolinearity)
                l_end_cols += adjusted_columns
                sd_end_cols += adjusted_columns
                l_elems += l_rows * adjusted_columns
                sd_elems += sd_rows  # BECAUSE NO MATTER HOW MANY UNIQUES (THUS COLUMNS), TOTAL ENTRIES ALWAYS ADD TO # ROWS
                # NOT ADJUSTING sd_elems FOR MULTICOLIN DROP NOR FOR PLACEHOLDER, TOO COMPLICATED FOR ESTIMATING PURPOSES
                self.l_mem += l_rows * adjusted_columns * (np_float if float_in_final_expanded else np_int)    # IF A FLOAT COLUMN IN FINAL, ALL COLUMNS WILL BE float64
                self.sd_mem += sd_rows * sd_int

            elif col_type == 'SPLIT_STR':
                l_end_cols += len(self.UNIQUES_HOLDER[col_idx])
                sd_end_cols += len(self.UNIQUES_HOLDER[col_idx])
                l_elems += l_rows * len(self.UNIQUES_HOLDER[col_idx])
                sd_elems += self.WORD_COUNT_HOLDER[col_idx]
                self.l_mem += l_rows * len(self.UNIQUES_HOLDER[col_idx]) * (np_float if float_in_final_expanded else np_int)    # IF A FLOAT COLUMN IN FINAL, ALL COLUMNS WILL BE float64
                self.sd_mem += self.WORD_COUNT_HOLDER[col_idx] * sd_int

            elif col_type in ['NNLM50']:
                l_end_cols += 50
                sd_end_cols += 50
                l_elems += l_rows * 50
                sd_elems += sd_rows * 50
                self.l_mem += l_rows * 50 * np_float    # IF HAS AN NNLM50 COLUMN, FINAL DATA MUST BE ALL FLOATS
                self.sd_mem += sd_rows * 50 * sd_float

            # PLACEHOLDER FOR POTENTIAL FUTURE TEXT ANALYTICS METHODS
            # elif col_type == 'TXT4': pass
            # elif col_type == 'TXT5': pass

        del adjusted_columns, col_type, np_float, np_int, sd_float, sd_int, float_in_final_expanded

        print_template = lambda _desc_, _rows_, _end_cols_, _elems_, _mem_: print \
            (f'{_desc_}'.ljust(50) + f' rows={_rows_:,}'.rjust(13) + \
             f' cols={_end_cols_:,}'.rjust(13) + f' elems={_elems_:,}'.rjust(20) + f' mem={int(_mem_):,} MB'.rjust(15))
        print_template(f'\nAPPROXIMATE SIZE EXPANDED AS LISTS:', int(l_rows), int(l_end_cols), int(l_elems), self.l_mem)
        print_template(f'APPROXIMATE SIZE EXPANDED AS SPARSE DICTS:', int(sd_rows), int(sd_end_cols), int(sd_elems), self.sd_mem)

        del l_rows, sd_rows, l_end_cols, sd_end_cols, l_elems, sd_elems, print_template  # DONT DELETE l_mem & sd_mem, NEED FOR MEM COMPARISON

        print(f'\nNUMBER OF DUMMY COLUMNS CONTRIBUTED BY CATEGORICAL COLUMNS:')
        if 'STR' in self.SUPPORT_OBJECTS[mdtypes_idx]:
            for modtype_idx in range(len(self.SUPPORT_OBJECTS[mdtypes_idx])):
                if self.SUPPORT_OBJECTS[mdtypes_idx][modtype_idx] == 'STR':
                    print(f'{self.SUPPORT_OBJECTS[msod.master_support_object_dict()["HEADER"]["position"]][modtype_idx][:48]}): '.ljust(50) +
                          f'{len(self.UNIQUES_HOLDER[modtype_idx]) - int(self.address_multicolinearity)}')
        else: print(f'NONE, NO CATEGORICAL COLUMNS.')

        print(f'\nNUMBER OF DUMMY COLUMNS CONTRIBUTED BY SPLIT STRING COLUMNS:')
        if 'SPLIT_STR' in self.SUPPORT_OBJECTS[mdtypes_idx]:
            for modtype_idx in range(len(self.SUPPORT_OBJECTS[mdtypes_idx])):
                if self.SUPPORT_OBJECTS[modtype_idx][modtype_idx] == 'SPLIT_STR':
                    print(f'{self.SUPPORT_OBJECTS[hdr_idx][modtype_idx][:48]}): '.ljust(50) +
                          f'{len(self.UNIQUES_HOLDER[modtype_idx])}')
        else: print(f'NONE, NO SPLIT STRING COLUMNS.')

        print(f'\nNUMBER OF COLUMNS CONTRIBUTED BY NNLM50 COLUMNS:')
        if 'NNLM50' in self.SUPPORT_OBJECTS[mdtypes_idx]:
            for modtype_idx in range(len(self.SUPPORT_OBJECTS[mdtypes_idx])):
                if self.SUPPORT_OBJECTS[mdtypes_idx][modtype_idx] == 'NNLM50':
                    print(f'{self.SUPPORT_OBJECTS[hdr_idx][modtype_idx][:48]}): 50')
            print(f'\nTOTAL NUMBER OF COLUMNS CONTRIBUTED BY NNLM50: {50 * np.sum(np.int8(self.SUPPORT_OBJECTS[mdtypes_idx]=="NNLM50"))}')
        else: print(f'NONE, NO NNLM50 COLUMNS.')

        print('*'*90 + f'\n')

        del hdr_idx, mdtypes_idx

        # END MEM USAGE ESTIMATES FOR EXPANSION AS ARRAY OR SPARSEDICT #######################################################
        ######################################################################################################################
        ######################################################################################################################



    def column_drop_iterator(self):
        '''Iteratively drop each column from passed object.'''
        # BEAR THIS FUNCTION WILL HAVE TO ACCOMMODATE DOES/DOESNT HAVE TARGET, WHAT STATS CAN BE GOT WILL CHANGE

        print(f'\n************************* '
              f'\nBEAR \nBYPASS \nwhole_data_object_stats \nUNTIL \nMLRegression \ncolumn_drop_iterator \nMODULE \nFINISHED'
              f'*************************\n')
        # self.column_drop_iterator(self.LEVELS, EXPANDED_COLUMN_NAMES, self.UNIQUES_HOLDER[col_idx], append_ones='Y')
        print()
        pass


    def whole_data_object_stats(self):
        '''Display statistics derived from MLRegression for passed object.'''
        print(f'\n************************* '
              f'\nBEAR \nBYPASS \nwhole_data_object_stats \n UNTIL MLRegression \ncolumn_drop_iterator \nMODULE \nFINISHED'
              f'*************************\n')

        # BEAR THIS FUNCTION WILL HAVE TO ACCOMMODATE DOES/DOESNT HAVE TARGET, WHAT STATS CAN BE GOT WILL CHANGE

        # if self.TARGET is None:.....
        # self.whole_data_object_stats(self.LEVELS, EXPANDED_COLUMN_NAMES, self.UNIQUES_HOLDER[col_idx], append_ones='Y')
        print()
        pass


    def context_update_for_column_drop(self, words):
        '''Context update for column drop during expansion of a STR column.'''
        # OVERWROTE IN CHILD
        self.CONTEXT_HOLDER.append(words)
        # BEAR DELETE IF ALL GOOD
        # self.CONTEXT_HOLDER.append(words)



        # BEAR DELETE ALL selfs THAT ARE NOT NEEDED



















if __name__ == '__main__':

    # LAST VERIFIED TEST CODE AND MODULE ON 7/2/23

    # 12/25/22 THIS TEST MODULE ONLY TESTS FOR FUNCTIONALITY OF PROMPTS/NOTES, MULTICOLIN CYCLER
    # (THINGS THAT ARENT USED BY ExpandCategoriesTestObjects)
    # TEST AUTODROPRIGHTMOST IN ExpandCategoriesTestObjects

    # TO TEST FOR ACCURACY OF OUTPUT OBJECT FOR INPUT CONTENTS/FORMAT/ORIENTATION/SIZE and OUTPUT FORMAT/ORIENTATION/SIZE,
    # USE THE TEST MODULE IN ExpandCategoriesTestObjects)

    dum_indicator = ' - '

    calling_module = gmn.get_module_name(str(sys.modules[__name__]))
    calling_fxn = 'guard_test'


    OBJECT_AS_LISTTYPE = [['A', 'B', 'A'], ['B', 'C', 'E'], ['F', 'F', 'G']]
    _rows = 3
    _columns = 3

    HEADER = np.array(['X', 'Y', 'Z'], dtype='<U1').reshape((1,-1))
    MODIFIED_DATATYPES = ['STR', 'STR', 'STR']
    BASE_SUPPORT_OBJECTS = msod.build_empty_support_object(3)

    # DONT PUT HEADER IN SUPPORT_OBJECTS HERE. DO IT IN THE LOOP BASED ON IF None OR NOT
    # BASE_SUPPORT_OBJECTS[msod.master_support_object_dict()["HEADER"]["position"]] = HEADER[0]

    BASE_SUPPORT_OBJECTS[msod.master_support_object_dict()["MODIFIEDDATATYPES"]["position"]] = MODIFIED_DATATYPES


    UNIQUES_AS_COLUMN = [['A','B'],['B','C','E'],['F','G']]
    UNIQUES_AS_ROW = [['A','B','F'], ['B','C','F'], ['A','E','G']]

    # W/O DROP RIGHTMOST
    ANSWER_OBJECT_COLUMN_ARRAY_RETURN_COLUMN = np.array([[1,0,1],[0,1,0],[1,0,0],[0,1,0],[0,0,1],[1,1,0],[0,0,1]])
    ANSWER_OBJECT_COLUMN_ARRAY_RETURN_ROW = np.array([[1, 0, 1, 0, 0, 1, 0], [0, 1, 0, 1, 0, 1, 0], [1, 0, 0, 0, 1, 0, 1]])
    ANSWER_OBJECT_ROW_ARRAY_RETURN_COLUMN = np.array([[1,0,0],[0,1,0],[0,0,1],[1,0,0],[0,1,0],[0,0,1],[1,0,0],[0,1,0],[0,0,1]])
    ANSWER_OBJECT_ROW_ARRAY_RETURN_ROW = np.array([[1,0,0,1,0,0,1,0,0],[0,1,0,0,1,0,0,1,0],[0,0,1,0,0,1,0,0,1]])
    ANSWER_HEADER_NONE_COLUMN = np.array([f'COLUMN1{dum_indicator}A', f'COLUMN1{dum_indicator}B', f'COLUMN2{dum_indicator}B', f'COLUMN2{dum_indicator}C', f'COLUMN2{dum_indicator}E', f'COLUMN3{dum_indicator}F', f'COLUMN3{dum_indicator}G'], dtype='<U100').reshape((1,-1))
    ANSWER_HEADER_NONE_ROW = np.array([f'COLUMN1{dum_indicator}A',f'COLUMN1{dum_indicator}B', f'COLUMN1{dum_indicator}F', f'COLUMN2{dum_indicator}B',f'COLUMN2{dum_indicator}C',f'COLUMN2{dum_indicator}F',f'COLUMN3{dum_indicator}A',f'COLUMN3{dum_indicator}E',f'COLUMN3{dum_indicator}G'], dtype='<U100').reshape((1,-1))
    ANSWER_HEADER_COLUMN = np.array([f'X{dum_indicator}A', f'X{dum_indicator}B', f'Y{dum_indicator}B', f'Y{dum_indicator}C', f'Y{dum_indicator}E', f'Z{dum_indicator}F', f'Z{dum_indicator}G'], dtype='<U100').reshape((1,-1))
    ANSWER_HEADER_ROW = np.array([f'X{dum_indicator}A',f'X{dum_indicator}B', f'X{dum_indicator}F', f'Y{dum_indicator}B',f'Y{dum_indicator}C',f'Y{dum_indicator}F',f'Z{dum_indicator}A',f'Z{dum_indicator}E',f'Z{dum_indicator}G'], dtype='<U100').reshape((1,-1))

    # W DROP RIGHTMOST
    ANSWER_OBJECT_COLUMN_ARRAY_RETURN_COLUMN_DROP_RIGHT = np.array([[1,0,1],[1,0,0],[0,1,0],[1,1,0]])
    ANSWER_OBJECT_COLUMN_ARRAY_RETURN_ROW_DROP_RIGHT = np.array([[1, 1, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0]])
    ANSWER_OBJECT_ROW_ARRAY_RETURN_COLUMN_DROP_RIGHT = np.array([[1,0,0],[0,1,0],[1,0,0],[0,1,0],[1,0,0],[0,1,0]])
    ANSWER_OBJECT_ROW_ARRAY_RETURN_ROW_DROP_RIGHT = np.array([[1,0,1,0,1,0],[0,1,0,1,0,1],[0,0,0,0,0,0]])
    ANSWER_HEADER_NONE_COLUMN_DROP_RIGHT = np.array([f'COLUMN1{dum_indicator}A', f'COLUMN2{dum_indicator}B', f'COLUMN2{dum_indicator}C', f'COLUMN3{dum_indicator}F'], dtype='<U100').reshape((1,-1))
    ANSWER_HEADER_NONE_ROW_DROP_RIGHT = np.array([f'COLUMN1{dum_indicator}A',f'COLUMN1{dum_indicator}B', f'COLUMN2{dum_indicator}B', f'COLUMN2{dum_indicator}C', f'COLUMN3{dum_indicator}A', f'COLUMN3{dum_indicator}E'], dtype='<U100').reshape((1,-1))
    ANSWER_HEADER_COLUMN_DROP_RIGHT = np.array([f'X{dum_indicator}A', f'Y{dum_indicator}B', f'Y{dum_indicator}C', f'Z{dum_indicator}F'], dtype='<U100').reshape((1,-1))
    ANSWER_HEADER_ROW_DROP_RIGHT = np.array([f'X{dum_indicator}A', f'X{dum_indicator}B', f'Y{dum_indicator}B', f'Y{dum_indicator}C', f'Z{dum_indicator}A', f'Z{dum_indicator}E'], dtype='<U100').reshape((1,-1))







    BYPASS_VALIDATION = [True, False]
    MASTER_GIVEN_ORIENTATION = ['COLUMN', 'ROW']
    MASTER_RETURN_ORIENTATION = ['COLUMN', 'ROW']
    MASTER_RETURN_FORMAT = ['PROMPT'] #, 'ARRAY', 'SPARSE_DICT']
    MASTER_ADDRESS_MULTICOLINEARITY = ['PROMPT'] # True,  True,  'PROMPT']
    MASTER_MULTICOLIN_CYCLER =        ['PROMPT'] # True,  True,  None ]
    MASTER_APPEND_ONES =              ['PROMPT'] # False, True,  None ]
    MASTER_AUTODROP =                 ['PROMPT'] # False, False, False]
    MASTER_HEADER = [HEADER, None]

    total_trials = np.product(list(map(len, (BYPASS_VALIDATION, MASTER_GIVEN_ORIENTATION, MASTER_RETURN_ORIENTATION, 
                                MASTER_RETURN_FORMAT, MASTER_ADDRESS_MULTICOLINEARITY, MASTER_HEADER))))

    ctr = 0
    for bypass_validation in BYPASS_VALIDATION:
        for given_orientation in MASTER_GIVEN_ORIENTATION:
            for return_orientation in MASTER_RETURN_ORIENTATION:
                for return_format in MASTER_RETURN_FORMAT:
                    for address_multicolinearity, cycler, append_ones, autodrop in zip(MASTER_ADDRESS_MULTICOLINEARITY, MASTER_MULTICOLIN_CYCLER, MASTER_APPEND_ONES, MASTER_AUTODROP):
                        for HEADER in MASTER_HEADER:
                            ctr += 1
                            print(f'\n')
                            print(f'*' * 90)
                            print(f'Running trial {ctr} of {total_trials}...')

                            SUPPORT_OBJECTS = BASE_SUPPORT_OBJECTS.copy()

                            if not HEADER is None:
                                SUPPORT_OBJECTS[msod.master_support_object_dict()["HEADER"]["position"]] = HEADER

                            DummyObject = ExpandCategoriesTemplate(
                                     OBJECT_AS_LISTTYPE,
                                     data_given_orientation=given_orientation,
                                     data_return_orientation=return_orientation,
                                     data_return_format=return_format,
                                     DATA_OBJECT_HEADER=HEADER,
                                     TARGET=None,
                                     target_given_orientation=None,
                                     TARGET_TRANSPOSE=None,
                                     target_transpose_given_orientation=None,
                                     TARGET_AS_LIST=None,
                                     target_as_list_given_orientation=None,
                                     target_is_multiclass=None,
                                     SUPPORT_OBJECT_AS_MDTYPES_OR_FULL_SUP_OBJ=SUPPORT_OBJECTS,
                                     address_multicolinearity=address_multicolinearity,
                                     auto_drop_rightmost_column=autodrop,
                                     multicolinearity_cycler=cycler,
                                     append_ones_for_cycler=append_ones,
                                     prompt_to_edit_given_mod_dtypes=False,
                                     print_notes=True,
                                     prompt_user_for_accept=True,
                                     bypass_validation=bypass_validation,
                                     calling_module='ExpandCategoriesTemplate',
                                     calling_fxn='guard_test')

                            act_calling_module = DummyObject.calling_module
                            act_calling_fxn = DummyObject.calling_fxn
                            act_given_orientation = DummyObject.data_given_orientation
                            act_given_format = DummyObject.data_given_format
                            act_return_orientation = DummyObject.data_return_orientation
                            act_return_format = 'SPARSE_DICT' if isinstance(DummyObject.DATA_OBJECT, dict) else 'ARRAY'
                            act_address_multicolinearity = address_multicolinearity
                            act_cycler = cycler
                            act_append_ones = append_ones
                            act_autodrop = DummyObject.auto_drop_rightmost_column
                            ACT_DATA_OBJECT = DummyObject.DATA_OBJECT
                            ACT_HEADER = DummyObject.SUPPORT_OBJECTS[msod.master_support_object_dict()["HEADER"]["position"]]
                            ACT_MODIFIED_DATATYPES = DummyObject.SUPPORT_OBJECTS[msod.master_support_object_dict()["MODIFIEDDATATYPES"]["position"]]
                            # ACT_DROPPED_COLUMN_NAMES = DummyObject.DROPPED_COLUMN_NAMES   # BEAR
                            ACT_CONTEXT_HOLDER = DummyObject.CONTEXT_HOLDER

                            # DONT MOVE THIS, MUST BE BEFORE DETERMINING EXP_DATA & EXP_HEADER
                            autodrop = act_autodrop     # TO COMPENSATE FOR USER ALLOWED TO CHANGE INSITU

                            # GET EXPECTED OBJECTS ##################################################################
                            if given_orientation == 'COLUMN':
                                if return_orientation == 'COLUMN':
                                    EXP_DATA_OBJECT = ANSWER_OBJECT_COLUMN_ARRAY_RETURN_COLUMN
                                    if autodrop is True: EXP_DATA_OBJECT = ANSWER_OBJECT_COLUMN_ARRAY_RETURN_COLUMN_DROP_RIGHT
                                elif return_orientation == 'ROW':
                                    EXP_DATA_OBJECT = ANSWER_OBJECT_COLUMN_ARRAY_RETURN_ROW
                                    if autodrop is True: EXP_DATA_OBJECT = ANSWER_OBJECT_COLUMN_ARRAY_RETURN_ROW_DROP_RIGHT
                            elif given_orientation == 'ROW':
                                if return_orientation == 'COLUMN':
                                    EXP_DATA_OBJECT = ANSWER_OBJECT_ROW_ARRAY_RETURN_COLUMN
                                    if autodrop is True: EXP_DATA_OBJECT = ANSWER_OBJECT_ROW_ARRAY_RETURN_COLUMN_DROP_RIGHT
                                elif return_orientation == 'ROW':
                                    EXP_DATA_OBJECT = ANSWER_OBJECT_ROW_ARRAY_RETURN_ROW
                                    if autodrop is True: EXP_DATA_OBJECT = ANSWER_OBJECT_ROW_ARRAY_RETURN_ROW_DROP_RIGHT

                            if not HEADER is None:
                                if given_orientation == 'COLUMN':
                                    EXP_HEADER = ANSWER_HEADER_COLUMN
                                    if autodrop is True: EXP_HEADER = ANSWER_HEADER_COLUMN_DROP_RIGHT
                                elif given_orientation == 'ROW':
                                    EXP_HEADER = ANSWER_HEADER_ROW
                                    if autodrop is True: EXP_HEADER = ANSWER_HEADER_ROW_DROP_RIGHT
                            elif HEADER is None:
                                if given_orientation == 'COLUMN':
                                    EXP_HEADER = ANSWER_HEADER_NONE_COLUMN
                                    if autodrop is True: EXP_HEADER = ANSWER_HEADER_NONE_COLUMN_DROP_RIGHT
                                elif given_orientation == 'ROW':
                                    EXP_HEADER = ANSWER_HEADER_NONE_ROW
                                    if autodrop is True: EXP_HEADER = ANSWER_HEADER_NONE_ROW_DROP_RIGHT

                            EXP_CONTEXT_HOLDER = []

                            exp_calling_module = calling_module
                            exp_calling_fxn = calling_fxn
                            exp_given_orientation = given_orientation
                            exp_return_orientation = return_orientation
                            exp_return_format = return_format if not return_format=='PROMPT' else act_return_format
                            exp_address_multicolinearity = act_address_multicolinearity #
                            exp_cycler = act_cycler                                     # TO COMPENSATE FOR USER ALLOWED TO CHANGE INSITU
                            exp_append_ones = act_append_ones                           #
                            exp_autodrop = autodrop                                     #
                            exp_columns = len(EXP_DATA_OBJECT) if return_orientation=='COLUMN' else len(EXP_DATA_OBJECT[0])
                            exp_rows = _rows
                            EXP_MODIFIED_DATATYPES = np.fromiter(('BIN' for _ in range(exp_columns)), dtype='<U20')

                            _insert = []
                            if autodrop:
                                if not HEADER is None:
                                    if given_orientation == 'ROW':
                                        _insert = [f'Deleted Z - G for multicolinearity auto_drop_rightmost_column',
                                                   f'Deleted Y - F for multicolinearity auto_drop_rightmost_column',
                                                   f'Deleted X - F for multicolinearity auto_drop_rightmost_column']
                                    elif given_orientation == 'COLUMN':
                                        _insert = [f'Deleted Z - G for multicolinearity auto_drop_rightmost_column',
                                                   f'Deleted Y - E for multicolinearity auto_drop_rightmost_column',
                                                   f'Deleted X - B for multicolinearity auto_drop_rightmost_column']
                                elif HEADER is None:
                                    if given_orientation == 'ROW':
                                        _insert = [f'Deleted COLUMN3 - G for multicolinearity auto_drop_rightmost_column',
                                                   f'Deleted COLUMN2 - F for multicolinearity auto_drop_rightmost_column',
                                                   f'Deleted COLUMN1 - F for multicolinearity auto_drop_rightmost_column']
                                    elif given_orientation == 'COLUMN':
                                        _insert = [f'Deleted COLUMN3 - G for multicolinearity auto_drop_rightmost_column',
                                                   f'Deleted COLUMN2 - E for multicolinearity auto_drop_rightmost_column',
                                                   f'Deleted COLUMN1 - B for multicolinearity auto_drop_rightmost_column']

                            if cycler:
                                pass
                                # CURRENTLY MANUALLY ENTERING TO NOT DROP ANY COLUMNS, SO NOTHING TO PUT IN CONTEXT_HOLDER

                            EXP_CONTEXT_HOLDER = np.hstack((EXP_CONTEXT_HOLDER, _insert))

                            if exp_return_format == 'SPARSE_DICT':
                                EXP_DATA_OBJECT = sd.zip_list_as_py_int(EXP_DATA_OBJECT)

                            # END GET EXPECTED OBJECTS ######################################################################


                            expected_results = \
                                (f"\nINCOMING DATA OBJECT IS A {type(OBJECT_AS_LISTTYPE)} AND HEADER IS " 
                                f"{'given' if not HEADER is None else None}, WITH {_rows} ROWS AND {_columns} " 
                                f"COLUMNS ORIENTED AS {given_orientation}.\n"
                                f"OBJECT SHOULD BE RETURNED AS A {exp_return_format} ORIENTED AS {return_orientation}.\n",
                                f"OBJECT_AS_LISTTYPE = 🤬\n",
                                f"DATA_OBJECT_HEADER = {HEADER}\n",
                                f"given_orientation = {given_orientation}\n",
                                f"return_orientation = {return_orientation}\n",
                                f"return_format = {return_format}\n",
                                f"address_multicolinearity = {address_multicolinearity}\n",
                                f"cycler = {cycler}\n",
                                f"append_ones = {append_ones}\n",
                                f"auto_drop_rightmost_column = {autodrop}\n",
                                f"bypass_validation = {bypass_validation}"
                            )

                            # OUTPUT VALIDATION ####################################################################

                            print(*expected_results)

                            NAMES = [
                                'calling_module',
                                'calling_fxn',
                                'given_orientation',
                                'return_orientation',
                                'return_format',
                                'exp_address_multicolinearity',
                                'exp_cycler',
                                'exp_append_ones',
                                'autodrop',
                                'DATA',
                                'HEADER',
                                'MODIFIED_DATATYPES',
                                'CONTEXT_HOLDER'
                            ]

                            EXPECTED_OUTPUTS = [
                                exp_calling_module,
                                exp_calling_fxn,
                                exp_given_orientation,
                                exp_return_orientation,
                                exp_return_format,
                                exp_address_multicolinearity,
                                exp_cycler,
                                exp_append_ones,
                                exp_autodrop,
                                EXP_DATA_OBJECT,
                                EXP_HEADER,
                                EXP_MODIFIED_DATATYPES,
                                EXP_CONTEXT_HOLDER
                            ]

                            ACTUAL_OUTPUTS = [
                                act_calling_module,
                                act_calling_fxn,
                                act_given_orientation,
                                act_return_orientation,
                                act_return_format,
                                act_address_multicolinearity,
                                act_cycler,
                                act_append_ones,
                                act_autodrop,
                                ACT_DATA_OBJECT,
                                ACT_HEADER,
                                ACT_MODIFIED_DATATYPES,
                                ACT_CONTEXT_HOLDER,
                            ]

                            print(f'ACT_CONTEXT_HOLDER LOOKS LIKE:')
                            print(ACT_CONTEXT_HOLDER)

                            for description, expected_thing, actual_thing in zip(NAMES, EXPECTED_OUTPUTS,
                                                                                 ACTUAL_OUTPUTS):
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
                                    #     raise Exception(f'\n*** TEST FAILED "==" METHOD ***\n')

                                if not is_equal:
                                    print(f'\n\n')
                                    print(f'*' * 90)
                                    print(f'Failed on trial {ctr} of at most {total_trials:,}')
                                    print(*expected_results)
                                    print()
                                    # print(f'\n\033[91mEXP_VALUE = \n{expected_thing}\033[0m\x1B[0m\n')
                                    # print(f'\n\033[91mACT_VALUE = \n{actual_thing}\033[0m\x1B[0m\n')
                                    print(f'\n\033[91mEXP_DATA_OBJECT = \n{EXP_DATA_OBJECT}\033[0m\x1B[0m\n')
                                    print(f'\n\033[91mACT_DATA_OBJECT = \n{ACT_DATA_OBJECT}\033[0m\x1B[0m\n')
                                    time.sleep(1)
                                    raise Exception(
                                        f'\n*** {description} FAILED EQUALITY TEST, \nexpected = \n{expected_thing}\n'
                                        f'actual = \n{actual_thing} ***\n')
                                else:
                                    pass  # print(f'\033[92m     *** {description} PASSED ***\033[0m\x1B[0m')

            # END OUTPUT VALIDATION ####################################################################

    print(f'\n\033[92m*** VALIDATION COMPLETED SUCCESSFULLY. ALL PASSED. ***\033[0m\x1B[0m\n')















































