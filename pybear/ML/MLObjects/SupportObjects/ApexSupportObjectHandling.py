import inspect, sys, time, warnings
from data_validation import validate_user_input as vui, arg_kwarg_validater as akv
from ML_PACKAGE._data_validation import ValidateObjectType as vot, list_dict_validater as ldv
from general_list_ops import list_select as ls
import numpy as np
import sparse_dict as sd
from debug import get_module_name as gmn
from MLObjects.PrintMLObject import print_object_preview as pop
from MLObjects.PrintMLObject.SmallObjectPreview import SmallObjectPreviewForASOH as sopfa
from MLObjects.SupportObjects import PrintSupportContents as psc, master_support_object_dict as msod



# DONT DO CONTEXT & KEEP!



# WHAT THIS MODULE NEEDS TO DO
# 1) BUILD SUPPORT OBJECTS FROM NOTHING (FILLED WITH DEFAULT VALUES)
# 2) BUILD SUPPORT OBJECTS FROM GIVEN SUPPORT_OBJECTS (TAKE IN ALREADY MADE)
# 3) BUILD SUPPORT OBJECTS FROM GIVEN DATA OBJECT
# 4) BUILD FULL SUPPORT_OBJECTS FROM ONLY GIVEN MODIFIED_DATATYPES (BACKFILL VAL DTYPES, autofill() OR defaultfill() OTHERS)
# 5) VALIDATE AGAINST DATA OBJECT AND OTHER SUPPORT OBJECTS


# 12/30/2022
# SUPPORT OBJECT HOLDER LAYOUT AS OF 1/3/23:
#                      |----- SLOT FOR EACH COLUMN IN DATA / TARGET / REF / TEST OBJECT -----|
# 0) HEADER         [[ STRINGS                                                                ]
# 1) VAL_DTYPES      [ 'STR', 'INT', 'BIN', 'FLOAT'                                           ]
# 2) MOD_DTYPES      [ 'INT', 'BIN', 'FLOAT', 'STR', 'SPLIT_STR', 'NNLM50'                    ]
# 3) FILTERING       [ [STRINGS], [], [], [],...                                              ]
# 4) MIN_CUTOFFS     [ INTEGER >= 0                                                           ]
# 5) USE_OTHER       [ 'Y', 'N'                                                               ]
# 6) START_LAG       [ INTEGER >= 0                                                           ]
# 7) END_LAG         [ INTEGER >= 0 & >= START_LAG                                            ]
# 8) SCALING         [ STRINGS                                                                ]]

# TO CHANGE THE ORDER, CHANGE IDXS IN self.master_dict()
# TO ADD A SUPPORT OBJECT, ADD TO self.master_dict(), CREATE A MODULE FOR IT, ADD TO BuildFullSupportObject,
# ADD RULES TO validate_allowed(), validate_against_objects()





# Apex IS PARENT OF Header, ValidatedDatatypes, QuickValidatedDatatypes, ModifiedDatatypes, Filtering, MinCutoff
# Header IS PARENT OF Scaling
# ValidatedDatatypes IS PARENT OF UseOther
# MinCutoff IS PARENT OF StartLag
# StartLag IS PARENT OF EndLag

#        | - Header             - | - Scaling
#        | - ValidatedDatatypes
# Apex - | - ModifiedDatatypes    (CANNOT BE A CHILD OF Validated)
#        | - Filtering
#        | - MinCutoff          - | - StartLag                  - | - EndLag
#        | - UseOther






class ApexSupportObjectHandle:
    '''BEAR put list of functions that can be called.'''
    def __init__(self,
                 OBJECT=None,
                 object_given_orientation=None,
                 columns=None,  # THE INTENT IS THAT THIS IS ONLY GIVEN IF OBJECT AND SUPPORT_OBJECT ARE NOT
                 OBJECT_HEADER=None,
                 SUPPORT_OBJECT=None,   # GIVEN AS [[]] OR [] IF SINGLE
                 prompt_to_override=False,
                 return_support_object_as_full_array=True,
                 bypass_validation=False,
                 calling_module=None,
                 calling_fxn=None):

        print(f'\n*** BEAR IS GOING INTO Apex.__init__() ***')

        self.this_module = gmn.get_module_name(str(sys.modules[__name__]))
        fxn = '__init__'
        self.calling_module = calling_module if not calling_module is None else self.this_module
        self.calling_fxn = calling_fxn if not calling_fxn is None else fxn



        self.VAL_TEXT_DTYPES = msod.val_text_dtypes()
        self.VAL_NUM_DTYPES = msod.val_num_dtypes()
        self.VAL_NUM_DTYPES_LIST = list(self.VAL_NUM_DTYPES.values())
        self.VAL_ALLOWED_DICT = self.VAL_NUM_DTYPES | self.VAL_TEXT_DTYPES
        self.VAL_ALLOWED_LIST = list(self.VAL_NUM_DTYPES_LIST) + list(self.VAL_TEXT_DTYPES.values())
        self.val_menu_allowed_cmds = ''.join(list(self.VAL_NUM_DTYPES.keys()) + list(self.VAL_TEXT_DTYPES.keys()))
        self.VAL_MENU_OPTIONS_PROMPT = [f'{self.VAL_ALLOWED_LIST[idx]}({self.val_menu_allowed_cmds[idx]})' for idx in
                                        range(len(self.VAL_ALLOWED_LIST))]
        self.VAL_REVERSE_LOOKUP = msod.val_reverse_lookup()


        self.MOD_TEXT_DTYPES = msod.mod_text_dtypes()  # , '4':'TXT4', '5':'TXT5'       DONT USE I,B,F
        self.MOD_NUM_DTYPES = msod.mod_num_dtypes()
        self.MOD_NUM_DTYPES_LIST = list(self.MOD_NUM_DTYPES.values())
        self.MOD_ALLOWED_DICT = self.MOD_NUM_DTYPES | self.MOD_TEXT_DTYPES
        self.MOD_ALLOWED_LIST = self.MOD_NUM_DTYPES_LIST + list(self.MOD_TEXT_DTYPES.values())
        self.mod_menu_allowed_cmds = ''.join(list(self.MOD_NUM_DTYPES.keys()) + list(self.MOD_TEXT_DTYPES.keys()))
        self.MOD_MENU_OPTIONS_PROMPT = [f'{self.MOD_ALLOWED_LIST[idx]}({self.mod_menu_allowed_cmds[idx]})' for idx in
                                    range(len(self.MOD_ALLOWED_LIST))]

        # PLACEHOLDERS FOR THINGS IN CHILDREN
        self.TEXT_DTYPES = None
        self.NUM_DTYPES = None
        self.NUM_DTYPES_LIST = None
        self.ALLOWED_DICT = None
        self.ALLOWED_LIST = None
        self.menu_allowed_cmds = None
        self.MENU_OPTIONS_PROMPT = None
        self.REVERSE_LOOKUP = None

        # END CENTRALIZED PLACE FOR MANAGING VDTYPES AND MDTYPES. ######################################################################

        ################################################################################################################
        ################################################################################################################
        # MANDATORY VALIDATION PART 1 - bypass_validation, OBJECT, HEADER, SUPPORT_OBJECT ##############################
        self.bypass_validation = akv.arg_kwarg_validater(bypass_validation, 'bypass_validation', [True, False, None],
                                                         self.this_module, fxn, return_if_none=False)
        # MUST KEEP THIS MANDATORY, GETS given_format, AND TURNS A LIST-TYPE OBJECT ENTERED AS [] TO NP [[]]
        #################################################################################################################
        # OBJECT, TURNS A LIST-TYPE OBJECT ENTERED AS [] TO NP [[]]
        self.given_format, self.OBJECT = ldv.list_dict_validater(OBJECT, 'OBJECT')
        #################################################################################################################
        # HEADER, TURNS A LIST-TYPE OBJECT ENTERED AS [] TO NP [[]]
        # IF RUNNING Header, HEADER CAN ONLY BE PASSED AS SUPPORT OBJECT, SO TO MAKE EVERYTHING CONSISTENT, ASSIGN IT TO OBJECT_HEADER ALSO
        if self.support_name() == 'HEADER':
            try: OBJECT_HEADER = SUPPORT_OBJECT.copy()
            except: pass

        header_format, self.OBJECT_HEADER = ldv.list_dict_validater(OBJECT_HEADER, 'OBJECT_HEADER')
        if header_format not in [None, 'ARRAY']:
            self._exception(fxn, f'GIVEN OBJECT_HEADER MUST BE A LIST-TYPE THAT CAN BE CONVERTED INTO NP ARRAY, OR None')
        del header_format
        #################################################################################################################
        # SUPPORT OBJECT, TURNS A LIST-TYPE OBJECT ENTERED AS [] TO NP [[]]
        # 1/4/23, ALL INCOMING SUPPORT_OBJECTS SHOULD BE ABLE TO HANDLE reshape, NOW INCLUDING Filtering BECAUSE PRE-super()
        # Filtering FORCES A FULLY-SIZED SUPPORT_OBJECT INTO super()
        support_format, self.SUPPORT_OBJECT = ldv.list_dict_validater(SUPPORT_OBJECT, 'SUPPORT_OBJECT')
        # self.SUPPORT_OBJECT COULD BE GIVEN AS A SINGLE VECTOR OR AN ARRAY OF ALL THE SUPPORT VECTORS, OR None
        if not support_format in [None, 'ARRAY']:
            self._exception(fxn, f'SUPPORT_OBJECT MUST BE A LIST-TYPE THAT CAN BE CONVERTED TO AN NPARRAY, OR None')

        if support_format == 'ARRAY' and len(self.SUPPORT_OBJECT) not in [1, len(self.master_dict())]:
            self._exception(fxn, f'IF SUPPORT_OBJECT IS GIVEN, IT MUST BE A SINGLE VECTOR OR A FULL ARRAY '
                                 f'WITH {len(self.master_dict())} ROWS')

        del support_format

        # IF SUPPORT OBJECT WAS GIVEN AS A SINGLE VECTOR ([]), KEEP AS [[]] AFTER ldv SO CAN
        # STANDARDIZE HANDLING WITH (SUPPORT OBJECT GIVEN AS FULL ARRAY)

        # END MANDATORY VALIDATION PART 1 - bypass_validation, OBJECT, HEADER, SUPPORT_OBJECT ##########################
        ################################################################################################################
        ################################################################################################################

        ################################################################################################################
        ################################################################################################################
        # OPTIONAL VALIDATION PART 2 - ARG / KWARG / SUPPORT ###########################################################
        if self.bypass_validation:
            self.object_given_orientation = object_given_orientation
            self.prompt_to_override = prompt_to_override
            self.return_support_object_as_full_array = return_support_object_as_full_array

        elif not self.bypass_validation:
            self.object_given_orientation = akv.arg_kwarg_validater(object_given_orientation, 'object_given_orientation',
                                                        ['COLUMN', 'ROW', None], self.this_module, fxn)
            self.prompt_to_override = akv.arg_kwarg_validater(prompt_to_override, 'prompt_to_override',
                                                        [True, False, None], self.this_module, fxn, return_if_none=False)
            self.return_support_object_as_full_array = akv.arg_kwarg_validater(return_support_object_as_full_array,
                'return_support_object_as_full_array', [True, False, None], self.this_module, fxn, return_if_none=False)
        # END OPTIONAL VALIDATION PART 2 - ARG / KWARG / SUPPORT ###############################################################
        ########################################################################################################################

        _a = not self.OBJECT is None
        _b = not self.SUPPORT_OBJECT is None
        _c = not self.OBJECT_HEADER is None
        _d = not self.object_given_orientation is None

        # if OBJECT is not None, orientation MUST BE GIVEN
        if _a and not _d: self._exception(fxn, f'IF OBJECT IS GIVEN, object_given_orientation MUST BE GIVEN')

        ########################################################################################################################
        ########################################################################################################################
        # GET columns, VALIDATE SHAPES #########################################################################################

        # THE CODE FOR (GETTING self.columns FROM THE OBJECTS) AND (VALIDATING OBJECT LENS) WOULD BE VIRTUALLY IDENTICAL.
        # SO IF ANY OBJECT(S) GIVEN MUST GET LEN(S), SO JUST DO VALIDATION ANYWAY. IF NO OBJECTS GIVEN, USE GIVEN columns
        # IF OBJECTS ARE GIVEN, OVERRIDE ANYTHING THAT MAY HAVE BEEN PUT IN columns KWARG WITH MEASURED VALUE
        # THE ORIGINAL INTENT IS columns IS TO BE GIVEN ONLY IF NO OBJECTS ARE GIVEN

        if not _a and not _b and not _c:   # IF NO OBJECTS WERE GIVEN, MUST USE columns TO BUILD

            if self.bypass_validation:
                self.columns = columns
            ########################################################################################################################
            # OPTIONAL VALIDATION PART 3 - GIVEN columns ######################################################################################
            elif not self.bypass_validation:
                if not columns: self._exception(fxn, f'columns MUST BE ENTERED IF NO OBJECTS ARE GIVEN')
                if not isinstance(columns, int) or columns < 1: self._exception(fxn, f'columns MUST BE AN INTEGER >= 1.')
                self.columns = columns
            # END OPTIONAL VALIDATION PART 3 - GIVEN columns ######################################################################################
            ########################################################################################################################

        # IF DATA OBJECT AND / OR HEADER AND / OR SUPPORT OBJECT ARE GIVEN
        else:
            oc, sc, hc = None, None, None
            if _a:  # DATA OBJECT GIVEN
                _ = self.object_given_orientation
                if self.given_format == 'ARRAY':
                    if _ == 'COLUMN': oc = len(self.OBJECT)
                    elif _ == 'ROW': oc = len(self.OBJECT[0])
                elif self.given_format == 'SPARSE_DICT':
                    if _ == 'COLUMN': oc = sd.outer_len(self.OBJECT)
                    elif _ == 'ROW': oc = sd.inner_len_quick(self.OBJECT)

            if _b: sc = len(self.SUPPORT_OBJECT[0])

            if _c: hc = len(self.OBJECT_HEADER[0])

            # IF ONLY ONE OF THREE
            if _a and not _b and not _c: self.columns = oc
            elif _b and not _a and not _c: self.columns = sc
            elif _c and not _a and not _b:
                self.columns = hc

            # IF TWO OF THREE OR ALL THREE
            elif _a + _b + _c >= 2:

                error_txt = lambda name1, len1, name2, len2: f'\nINCONGRUENT NUMBER OF COLUMNS FOR {name1} ({len1}) AND {name2} ({len2})'
                ####################################################################################################################
                # MANDATORY VALIDATION PART 4 - EQUAL LENS ############################################################################
                if _a and _b:
                    if oc != sc: self._exception(fxn, error_txt('OBJECT', oc, 'SUPPORT_OBJECT', sc))
                    self.columns = oc
                if _a and _c:
                    if oc != hc: self._exception(fxn, error_txt('OBJECT', oc, 'OBJECT_HEADER', hc))
                    self.columns = oc
                if _b and _c:
                    if sc != hc: self._exception(fxn, error_txt('SUPPORT_OBJECT', sc, 'OBJECT_HEADER', hc))
                    self.columns = sc
                # END MANDATORY VALIDATION PART 4A - EQUAL LENS ############################################################################
                ####################################################################################################################
                del error_txt

            del oc, sc, hc

        del _d  # KEEP _a _b AND _c

        # NOW KNOW # columns AND HAVE ALL OBJECTS IF GIVEN, OTHERWISE OBJECTS NOT GIVEN ARE None
        # END GET columns, VALIDATE SHAPES #####################################################################################
        ########################################################################################################################
        ########################################################################################################################

        self.allowed_values()
        self.empty_value = msod.empty_value()    # VALUE THAT FILLS A NEWLY CREATED "EMPTY" SUPPORT_OBJECT

        # BUILD HOLDERS FOR QUICK_VAL_DTYPES, MOD_DTYPES_HOLDER
        self.QUICK_VAL_DTYPES = np.full(self.columns, self.empty_value)
        self.MOD_DTYPES_HOLDER = np.full(self.columns, self.empty_value)

        if not self.OBJECT_HEADER is None:
            self.QUICK_HEADER = self.OBJECT_HEADER
        elif self.support_name().upper() == 'HEADER' and not self.SUPPORT_OBJECT is None:
            if len(self.SUPPORT_OBJECT) == 1:
                self.QUICK_HEADER = self.SUPPORT_OBJECT[0].reshape((1,-1))
            elif len(self.SUPPORT_OBJECT) == len(self.master_dict()):
                self.QUICK_HEADER = self.SUPPORT_OBJECT[self.master_dict()['HEADER']['position']].reshape((1,-1))
        else:
            self.QUICK_HEADER = np.fromiter((f'COLUMN{_+1}' for _ in range(self.columns)), dtype='<U15').reshape((1,-1))

        ########################################################################################################################
        ########################################################################################################################
        # HANDLING FOR SUPOBJ REGARDLESS IF DATA OBJECT IS GIVEN ###############################################################
        # FIND OUT IF SUPOBJ IS/ISNT GIVEN; IF NOT GIVEN, BUILD AN EMPTY SINGLE, IF GIVEN, FIND OUT WHAT SUPPORTS ARE FULL/EMPTY

        # OVERWRITE W is_empty_getter BELOW IF SUPOBJ IS GIVEN AND FULL
        self.header_is_empty = True
        self.vdtypes_is_empty = True
        self.mdtypes_is_empty = True
        self.filtering_is_empty = True
        self.mincutoffs_is_empty = True
        self.useother_is_empty = True
        self.startlag_is_empty = True
        self.endlag_is_empty = True
        self.scaling_is_empty = True
        self.qvdtypes_is_empty = True


        if not _b:    # IF SUPPORT_OBJECT WAS NOT GIVEN, CREATE AS SINGLE ([[]]) AND SET actv_idx TO 0`
            self.SUPPORT_OBJECT = np.full((1, self.columns), self.empty_value, dtype=object)
            self.actv_idx = 0
            self.supobj_is_full = False
            _b = True
            active_is_empty = True
            # ALL THE is_empties FROM ABOVE STAY True
        elif _b:
            if len(self.SUPPORT_OBJECT) == 1:
                self.actv_idx = 0
                self.supobj_is_full = False
            elif len(self.SUPPORT_OBJECT) == len(self.master_dict()):
                self.set_active_index()
                self.supobj_is_full = True

            '''
            12/30/22
            IT IS POSSIBLE THAT SUPPORT_OBJECT IS GIVEN AS PART OF LARGER ARRAY, BUT ISNT REALLY "GIVEN" PER SE (EVEN AS A 
            SINGLE, FOR THAT MATTER, IF USER DELIBERATELY CREATES AND PASSES A SINGLE VECTOR FULL OF empty_value WHICH THEY 
            SHOULDNT, WHY BUILD AN EMPTY SINGLE OBJECT W NO USEFUL INFORMATION AND PASS IT?).
            "EMPTY" STATE MEANS ROW IN QUESTION IS FULL OF self.empty_value. MUST HAVE BEEN CREATED AS PART OF FULL ARRAY BY 
            PASSING THRU A HANDLER CLASS OTHER THAN ITS OWN, OR USER CREATED THAT WAY AND PASSED IT, EITHER WAY IS "NOT GIVEN".
            "GIVEN" STATE MEANS ROW IN QUESTION PASSED THRU ITS OWN HANDLER CLASS, CAUSING IT TO NOT BE empty-value, BUT
            default_value() OR SOMETHING ELSE THAT THE USER ENTERED, EITHER BY PASSING AN APPROPRIATELY FILLED OBJECT, OR MAKING
            EDITS TO IT WHILE IN ITS HANDLER CLASS.
            '''

            # IF A SUPPORT_OBJECT IS GIVEN
            #   - EDIT IN-PLACE, USING THE LOCATION IN active_idx() IF FULL, OR 0 IF SINGLE
            #   - IF MANAGE FINAL SUPOBJ RETURN SIZE LAST

            if not self.supobj_is_full:
                self.actv_idx = 0
                # ALL THE is_empties FROM ABOVE STAY True
                # THE ACTIVE STILL COULD BE EMPTY
            elif self.supobj_is_full:
                # actv_idx SET BY set_active_index() MANY LINES ABOVE HOLDS
                if _c:  # IF HEADER WAS GIVEN SEPARATELY, APPLY TO FULL SUPPORT OBJECT, AND SAY HEADER IS NOT EMPTY
                    # THIS APPLIES IN ANY CHILD THAT CALLS Apex AS super()
                    self.SUPPORT_OBJECT[self.master_dict()['HEADER']['position']] = self.OBJECT_HEADER[0]
                    self.header_is_empty = False
                elif not _c:  # IF A HEADER WAS NOT GIVEN, LOOK TO SEE IF SUPPORT OBJECT ALREADY HAS A HEADER IN IT
                    # IF IS NOT EMPTY CAN OVERRIDE IN Header OR A GIVEN HEADER CAN BE PASSED TO ANY INDIVIDUAL SUPPORT
                    # OBJECT CHILD CLASS
                    # IF IS EMPTY WILL default_fill() RIGHT HERE OR IN Header
                    self.header_is_empty = self.is_empty_getter(self.master_dict()['HEADER']['position'])
                    if self.header_is_empty:
                        self.SUPPORT_OBJECT[self.master_dict()['HEADER']['position']] = \
                            np.fromiter((f'COLUMN{_ + 1}' for _ in range(self.columns)), dtype=object)
                        self.header_is_empty = False

                # GET THESE BEFORE edit_menu() and validate_allowed(), THEY MAY TAKE YOU TO PLACES THAT HAVE is_empties DEPENDENCIES
                self.vdtypes_is_empty = self.is_empty_getter(self.master_dict()['VALIDATEDDATATYPES']['position'])
                self.mdtypes_is_empty = self.is_empty_getter(self.master_dict()['MODIFIEDDATATYPES']['position'])
                self.filtering_is_empty = self.is_empty_getter(self.master_dict()['FILTERING']['position'])
                self.mincutoffs_is_empty = self.is_empty_getter(self.master_dict()['MINCUTOFFS']['position'])
                self.useother_is_empty = self.is_empty_getter(self.master_dict()['USEOTHER']['position'])
                self.startlag_is_empty = self.is_empty_getter(self.master_dict()['STARTLAG']['position'])
                self.endlag_is_empty = self.is_empty_getter(self.master_dict()['ENDLAG']['position'])
                self.scaling_is_empty = self.is_empty_getter(self.master_dict()['SCALING']['position'])

            active_is_empty = self.is_empty_getter(self.actv_idx)


        # IF self.SUPPORT_OBJECT WAS NOT GIVEN, HAS BEEN CREATED AS EMPTY SINGLE, AND ALL _is_givens ARE STILL FALSE.
        # IF GIVEN AS SINGLE OR FULL NOW KNOW WHAT IS/ISNT EMPTY.

        # END HANDLING FOR SUPOBJ REGARDLESS IF DATA OBJECT IS GIVEN ###########################################################
        ########################################################################################################################
        ########################################################################################################################

        # NOW HAVE self.columns, KNOW IF OBJECT IS / ISNT GIVEN (_a CAN BE True OR False), actv_idx HAS BEEN ASSIGNED,
        # AND self.SUPPORT_OBJECT EXISTS, WHETHER AS GIVEN OR IN EMPTY STATE (_b MUST BE True).
        # MANAGE FOR WHEN:
        # self.active_is_emtpy is False and OBJECT IS GIVEN (_a is True) OR NOT GIVEN (_a is False)
        # self.active_is_empty is True and OBJECT IS GIVEN (_a is True) OR NOT GIVEN (_a is False)

        if not active_is_empty:
            # IF OBJECT IS RECEIVED IN A NON-EMPTY STATE, EITHER IT PASSED THRU ITS HANDLER CLASS OR USER MADE IT THAT WAY,
            # SO IT MUST BE "GIVEN". IF OBJECT HAS ILLEGAL VALUES, USER MADE IT THAT WAY.
            # OVERWRITE IF HAS ILLEGAL VALUES OR IF USER OPTS TO EDIT
            # THIS HANDLES BOTH IF _a AND not _a

            self.fill_from_kwarg()

            if self.bypass_validation:
                # IF NO VALIDATION, ALLOW EDIT IF USER WANTED IT
                if self.prompt_to_override:
                    if vui.validate_user_str(f'Edit given {self.support_name()}? (y/n) > ', 'YN') == 'Y':
                        self.edit_menu(fxn=fxn)
            elif not self.bypass_validation:
                ######################################################################################################################
                # OPTIONAL VALIDATION 5 - CONTENTS OF SUPPORT_OBJECT[actv_idx] ######################################################

                # FORCE OVERWRITE IF HAS ILLEGAL VALUES, BUT IF PASSES AND USER WANTS EDIT THEN ALLOW
                self.validate_allowed(kill=not self.prompt_to_override, fxn=fxn)
                self.validate_against_objects(kill=not self.prompt_to_override, fxn=fxn)

                if self.prompt_to_override:   # IF THERE WAS AN ERROR THEN USER GOT TO DO OVERRIDES
                    if vui.validate_user_str(f'Edit given {self.support_name()}? (y/n) > ', 'YN') == 'Y':
                        self.edit_menu(fxn=fxn)

                # END OPTIONAL VALIDATION 5 - CONTENTS OF SUPPORT_OBJECTS[actv_idx] ########################################
                ############################################################################################################

            # LEN VALIDATION ALREADY DONE ABOVE
            # prompt_to_override ON GIVEN actv_idx VALUES WAS OFFERED IF USER WANTED IT
            # VALIDATION WAS DONE ABOVE ON actv_idx IF NOT EMPTY AND IF bypass_validation IS False, FIX WAS FORCED IF ERROR,
            #      MEANING ALL VALID ENTRIES IN active index

        elif active_is_empty:
            # if OBJECT NOT GIVEN

            if not _a: self.default_fill()   # CREATE SUPPORT OBJECT ONLY W DEFAULT VALUE

            # if OBJECT IS GIVEN:F
            elif _a: self.autofill()    # FILL SUPPORT_OBJECT[actv_idx] W SPECIAL INSTRUCTIONS IN CHILD

            # NO VALIDATION, default_fill() AND autofill() EXPECTED TO FILL CORRECTLY

            if self.prompt_to_override:
                if vui.validate_user_str(f'\nOverride {"autofill" if _a else "default"} {self.support_name()} values? (y/n) > ',
                                         'YN') == 'Y':

                    self.edit_menu(fxn=fxn, terminate_text=f'USER TERMINATED OVERRIDING FILL OF EMPTY SUPOBJ.')

            active_is_empty = False

        # AT THIS POINT, WHATEVER MODULE THIS IS RUNNING IN, SUPPORT_OBJECT[self.actv_idx] IS NOW FULL, SO UPDATE WHICHEVER _is_empty IT IS
        if 'HEADER' in self.support_name().upper(): self.header_is_empty = False
        elif 'VALIDATED' in self.support_name().upper(): self.vdtypes_is_empty = False
        elif 'MODIFIED' in self.support_name().upper(): self.mdtypes_is_empty = False
        elif 'FILTERING' in self.support_name().upper(): self.filtering_is_empty = False
        elif 'CUTOFF' in self.support_name().upper(): self.mincutoffs_is_empty = False
        elif 'OTHER' in self.support_name().upper(): self.useother_is_empty = False
        elif 'START' in self.support_name().upper(): self.startlag_is_empty = False
        elif 'END' in self.support_name().upper(): self.endlag_is_empty = False
        elif 'SCALING' in self.support_name().upper(): self.scaling_is_empty = False

        del _a, _b, _c, active_is_empty


        # FINISH OFF self.SUPPORT_OBJECT BASED ON HOW/IF IT WAS GIVEN AND USER INPUT FOR return_support_object_as_full_array KWARG
        # ACTIVE INDEX MUST BE FULL AT THIS POINT

        # IF SUPPORT_OBJECT WAS NOT GIVEN, WAS BUILT AS SINGLE [[]] AND self.actv_idx WAS HARD SET TO 0
        # IF WAS GIVEN AS SINGLE, WAS RESHAPED TO [[]] AND self.actv_idx WAS HARD SET TO 0
        # IF WAS GIVEN AS FULL, self.actv_idx WAS SET BY self.set_active_index()

        if self.return_support_object_as_full_array:
            if self.supobj_is_full: pass  # actv_idx STAYS WHAT IT WAS SET TO BY set_active_index()
            elif not self.supobj_is_full:
                DUM_SUPPORT = msod.build_empty_support_object(self.columns)
                DUM_SUPPORT[self.master_dict()[self.support_name().upper()]['position']] = self.SUPPORT_OBJECT[0]
                self.SUPPORT_OBJECT = DUM_SUPPORT; del DUM_SUPPORT
                # IF HEADER WAS GIVEN, APPLY IT
                if not self.OBJECT_HEADER is None:
                    self.SUPPORT_OBJECT[self.master_dict()['HEADER']['position']] = self.OBJECT_HEADER[0].copy()
                elif self.OBJECT_HEADER is None:
                    self.SUPPORT_OBJECT[self.master_dict()['HEADER']['position']] = \
                        np.fromiter((f'COLUMN{_+1}' for _ in range(self.columns)), dtype=object)

                self.supobj_is_full = True
                self.set_active_index()

        elif not self.return_support_object_as_full_array:
            if self.supobj_is_full:
                self.SUPPORT_OBJECT = self.SUPPORT_OBJECT[self.actv_idx]
                self.supobj_is_full = False
                self.actv_idx = 0
            elif not self.supobj_is_full:
                self.SUPPORT_OBJECT = self.SUPPORT_OBJECT[0]
                # actv_idx STAYS 0

        print(f'\n*** BEAR IS LEAVING Apex.__init__() ***')

    # END __init__ #########################################################################################################
    ########################################################################################################################
    ########################################################################################################################
    ########################################################################################################################


    ##########################################################################################################################
    ##########################################################################################################################
    # OVERWRITTEN IN CHILDREN ###############################################################################################

    def support_name(self):
        '''Name of child's support object.'''
        # OVERWRITE IN CHILD
        return 'HEADER'    # JUST TO SPOOF Apex

    def default_value(self):
        '''Default value to fill support object.'''
        # OVERWRITE IN CHILD
        return 'None'

    def allowed_values(self):
        'Allowed values for validation.'''
        # OVERWRITE IN CHILD
        pass

    def autofill(self):
        '''Unique method to fill particular support object.'''
        # OVERWRITE IN CHILD
        self.default_fill()

    # END OVERWRITTEN IN CHILDREN ############################################################################################
    ##########################################################################################################################
    ##########################################################################################################################


    def set_active_index(self):
        '''Allows support object to be uniquely set in children to a single object or row of a larger object. '''
        self.actv_idx = self.master_dict()[self.support_name().upper()]['position']   # NOT OVERWRIT IN ANY CHILD


    def master_dict(self):
        return msod.master_support_object_dict()


    def _exception(self, fxn, words):
        '''Exception raise and error print-out for this module.'''
        raise Exception(f'\n*** {self.support_name()}.{fxn}() called via {self.calling_module}.{self.calling_fxn}() >>> '
                        f'{words}')


    def get_fxn(self, fxn, inspect_stack):
        return fxn if not fxn is None else inspect_stack


    def is_empty_getter(self, _idx):
        return msod.is_empty_getter(supobj_idx=_idx, supobj_name=None,
                             SUPOBJ=self.SUPPORT_OBJECT, calling_module=self.calling_module, calling_fxn=self.calling_fxn)


    def special_supobj_handling(self):
        pass


    def vdtypes(self, quick=None, print_notes=None):
        '''Unique method to fill validated datatypes in ValidatedDatatypes and QuickValidatedDatatypes.'''
        # 1/10/23 IN ORDER TO RECYCLE THIS CODE, THIS MUST BE IN Apex AND CALLED INTO autofill() IN ValidatedDatatypes AND
        # QuickValidatedDatatypes. THIS CANNOT BE IN ValidatedDatatypes THEN Quick MADE A CHILD OF ValidatedDatatypes BECAUSE
        # Quick IS CALLED IN ValidatedDatatypes CAUSING A CIRCULAR IMPORT.  IN ORDER TO GET THOSE CLASSES TO STAND ALONE, BUT
        # SHARE THIS CODE, BOTH MUST BE CHILDREN OF Apex THEN THIS CODE ALSO BEING IN Apex IS SHARED BETWEEN THE TWO.

        fxn = inspect.stack()[0][3]

        if self.support_name() == 'VALIDATEDDATATYPES':
            quick = akv.arg_kwarg_validater(quick, 'quick', [True, False, None], self.this_module, fxn)
            print_notes = akv.arg_kwarg_validater(print_notes, 'print_notes', [True, False, None], self.this_module, fxn)
        else:
            quick = akv.arg_kwarg_validater(quick, 'quick', [True, False], self.this_module, fxn)
            print_notes = akv.arg_kwarg_validater(print_notes, 'print_notes', [True, False], self.this_module, fxn)

        # "None" CAN ONLY BE PASSED TO vdtypes & print_notes IF THE MODULE IS ValidatedDatatypes.  THESE ARE SENTINELS
        # THAT REQUIRES HARD KWARG IN EVERY OTHER MODULE BUT IN ValDtypes THE None KWARG ALLOWS ACCESS TO
        # self.quick_vdtypes AND self.print_notes, WHICH ARE KWARGS ONLY IN THE ValidatedDatatypes CHILD CLASS.
        quick = self.quick_vdtypes if quick is None else quick
        print_notes = self.print_notes if quick is None else print_notes

        ##############################################################################################################
        # GET / OVERRIDE / ACCEPT VALIDATED DTYPES ####################################################################

        # IF SPARSE_DICT, TRANSPOSE AHEAD OF TIME TO GET INTO COLUMN
        if self.given_format == 'SPARSE_DICT' and self.object_given_orientation == 'ROW':
            self.OBJECT = sd.core_sparse_transpose(self.OBJECT)

        if print_notes: print(f'\nGetting column datatypes...')

        for col_idx in range(self.columns):

            zz = self.QUICK_HEADER[0][col_idx]
            __ = self.object_given_orientation

            if self.given_format == 'ARRAY':
                if __ == 'COLUMN': DUM_DATA = self.OBJECT[col_idx]
                elif __ == 'ROW': DUM_DATA = self.OBJECT[:, col_idx]
            elif self.given_format == 'SPARSE_DICT':
                # 'ROW' WAS TRANSPOSED ABOVE FOR SPEED, AVOID core_multi_select_inner
                DUM_DATA = np.fromiter(self.OBJECT[col_idx].values(), dtype=np.float64)
                DUM_DATA = np.insert(DUM_DATA, 0, 0, axis=0)

            if quick is False: SAMPLE = DUM_DATA
            elif quick is True: SAMPLE = np.random.choice(DUM_DATA, 500, replace=True)

            if print_notes: print(f'Reading {zz}...')
            try:
                # 1/5/23 THIS RESULT IS EXACT AND CAN GO IN SUPPORT_OBJECT AND NUMBER TYPES INTO MOD_DTYPES
                # BEAR THIS IS TO GO FAST WHEN DEBUGGING
                # UNHASH THIS
                validated_datatype = vot.ValidateObjectType(SAMPLE).ml_package_object_type()
                # DELETE THIS
                # validated_datatype = 'FLOAT'
                # ONLY SET QUICK_VAL_DTYPES, FILL OTHER VAL/MOD DTYPE HOLDERS OUTSIDE OF THIS FXN

                # ACCOMMODATE THAT vdtypes CAN ACT ON A CLASS AFTER init IS COMPLETED.  NORMALLY IN ValidatedDatatypes
                # vdtypes WOULD ACT ON SUPOBJ THRU autofill IN init, AND ALWAYS SEES SUPOBJ IN [[]] STATE IN init. BUT
                # quick_val_dtypes IS USING vdtypes BY ACTING ON AN INSTANCE OF Apex AFTER init OF IT IS FINISHED AND
                # SUPOBJ IS RETURNED AS SINGLE. OBJECT IS [[]] IF RETURNING AS FULL, BUT IS [] IF RETURNING AS SINGLE,
                # BASED ON WHAT return_support_object_as_full_array SAYS
                if len(self.SUPPORT_OBJECT.shape)==2:
                    self.SUPPORT_OBJECT[self.actv_idx][col_idx] = validated_datatype
                elif len(self.SUPPORT_OBJECT.shape)==1:
                    self.SUPPORT_OBJECT[col_idx] = validated_datatype

            except:
                error_msg = f'ValidateObjectType GAVE EXCEPTION WHEN TRYING TO IDENTIFY COLUMN {zz} DURING READ OF VALIDATED DATATYPES'
                print(f'\n*** {error_msg} ***\n')
                # SHOW WHAT DATA IS IN BAD COLUMN AND GIVE USER OPTIONS
                self.small_preview_object(idx=col_idx)
                # ONLY ALLOWED SINGLE EDIT, PRINT OUTS
                self.edit_menu(allowed='ESIDBCXT', fxn=fxn, terminate_text=error_msg)

                del error_msg

        del zz, __, DUM_DATA, SAMPLE

        if print_notes: print(f'Done getting column datatypes.')

        # TRANSPOSE BACK AFTER GETTING V_DTYPES
        if self.given_format == 'SPARSE_DICT' and self.object_given_orientation == 'ROW':
            self.OBJECT = sd.core_sparse_transpose(self.OBJECT)

        # END GET / OVERRIDE / ACCEPT VALIDATED DTYPES ################################################################
        ##############################################################################################################


    def get_quick_val_dtypes(self):
        # 1/15/23 THIS MUST STAND ALONE, DO NOT PUT UNDER validate_against_objects(), WILL NEED TO BE AN Apex METHOD IF
        # EVER validate_against_objects() IS PARTED OUT AMONGST THE CHILDREN.

        # IMPOSSIBLE FOR QUICK_VAL_DTYPES TO NOT EQUAL VAL_DTYPES WHEN VAL_DTYPES IS GIVEN, BECAUSE IF VAL_DTYPES IS GIVEN,
        # QUICK IS MADE FROM IT. IF VAL_DTYPES DOES NOT EXIST, THEN GENERATED QUICK VD COULD DIFFER FROM A GENERATED FULL VD.

        # ONLY WAY TO GET HERE IS not self.bypass_validation, SO DONT DO ANY VALIDATION ifS BECAUSE VALIDATION IS ON

        print(f'\n\n\n\033[94m*** BEAR IN get_quick_val() *** \033[0m\n\n\n')

        fxn = inspect.stack()[0][3]

        if self.OBJECT is None:
            # CANT VALIDATE AGAINST OBJECT IF IT DOESNT EXIST
            # QUICK_VAL_DTYPES STAYS EMPTY FROM __init__
            # qvdtypes_is_empty STAYS True
            pass
        elif not self.OBJECT is None:
            if not self.qvdtypes_is_empty:
                # JUST KEEP IT AS IT IS IF ALREADY SOMETHING IN IT. IF EVER "REAL" ValidatedDatatypes ARE GENERATED
                # POST-FACTO VIA autofill() (ONLY WAY THEY COULD BE GENERATED IS BY autofill()), QUICK_VAL_DTYPES IS
                # OVERWRITTEN WITH THESE VALUES IMMEDIATELY AFTER autofill().
                pass
            elif self.qvdtypes_is_empty:
                # 1/12/23 BELIEVE THIS WHOLE METHOD IS ONLY ACCESSED IF self.bypass_validation IS False
                # GOING TO GENERATE QUICK VAL DTYPES NO MATTER WHAT, TO VERIFY VAL_DTYPES IF GIVEN, OR TO GET THEM NEW OTHERWISE, SO DO IT HERE

                DumValDtypes = ApexSupportObjectHandle(OBJECT=self.OBJECT,
                                          object_given_orientation=self.object_given_orientation,
                                          columns=self.columns,
                                          OBJECT_HEADER=self.QUICK_HEADER,
                                          SUPPORT_OBJECT=None,  # GIVEN AS [[]] OR [] IF SINGLE
                                          prompt_to_override=False, #self.prompt_to_override,
                                          return_support_object_as_full_array=False,
                                          bypass_validation=True,
                                          calling_module=self.this_module,
                                          calling_fxn=fxn)

                DumValDtypes.vdtypes(quick=True, print_notes=False)
                self.QUICK_VAL_DTYPES = DumValDtypes.SUPPORT_OBJECT
                del DumValDtypes


    def get_active_mod_dtypes(self):
        # 1/15/23 THIS MUST STAND ALONE, DO NOT PUT UNDER validate_against_objects(), WILL NEED TO BE AN Apex METHOD IF
        # EVER validated_against_objects() IS PARTED OUT AMONGST THE CHILDREN.

        # DONT KNOW IF validate_against_objects() COULD BE CALLED IF MDTYPES_is_empty is False AND mdtypes_is_empty is True
        # (SEE ValidatedDatatypes), SO HEDGE BETS AND BUILD A MOD_DTYPED HOLDER OBJECT. mdtypes_is_empty IS FOR MOD DTYPES
        # PASSED VIA SUPPORT_OBJECT, MDTYPES_is_empty IS FOR MOD DTYPES PASSED VIA "MODIFIED_DATATYPES" KWARG IN
        # ValidatedDatatypes CHILD. THINKING IT ALSO MATTERS WHETHER IN MODIFIEDDATATYPES MODULE OR NOT, BECAUSE IN THERE
        # MODIFIED_DATATYPES COULD EXIST not is_empty AND not supobj_id_full SO actv_idx IS 0, CAUSING master_dict() TO BE INVALID.
        if not self.support_name().upper() not in ['MODIFIEDDATATYPES', 'VALIDATEDDATATYPES']:
            # MOD DTYPES COULD ONLY BE AVAILABLE THRU FULL SUPPORT_OBJECTS, MDTYPES COULD NOT BE AVAILABLE VIA KWARG
            if not self.mdtypes_is_empty:
                self.MOD_DTYPES_HOLDER = self.SUPPORT_OBJECT[self.master_dict()['MODIFIEDDATATYPES']['position']].copy()
            # else self.MOD_DTYPES_HOLDER STAYS AS self.empty_value
        elif self.support_name().upper() == 'MODIFIEDDATATYPES':  # MOD DTYPES MUST BE AVAILABLE VIA self.actv_idx
            if not self.mdtypes_is_empty: self.MOD_DTYPES_HOLDER = self.SUPPORT_OBJECT[self.actv_idx].copy()
            # else self.MOD_DTYPES_HOLDER STAYS AS self.empty_value
        elif self.support_name().upper() == 'VALIDATEDDATATYPES':
            # IF MOD DTYPES ARE AVAILABLE VIA KWARG, THAT WOULD OVERWRITE MOD DTYPES GIVEN VIA SUPPORT OBJECT, SO CHECK THAT FIRST
            if not self.MDTYPES_is_empty: self.MOD_DTYPES_HOLDER = self.MODIFIED_DATATYPES.copy()
            elif not self.mdtypes_is_empty:
                self.MOD_DTYPES_HOLDER = self.SUPPORT_OBJECT[self.master_dict()['MODIFIEDDATATYPES']['position']].copy()
            # else self.MOD_DTYPES_HOLDER STAYS AS self.empty_value

        if not self.MOD_DTYPES_HOLDER is None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                is_expanded = True not in np.fromiter((_ in self.MOD_TEXT_DTYPES.values() for _ in self.MOD_DTYPES_HOLDER), dtype=bool)
        else:
            # IF MOD DTYPES IS NOT AVAILABLE, USE THE LEAST RESTRICTIVE STATE, THINKING THAT IS "is_expanded is True"
            # E.G. THIS WOULD ALLOW MIN CUTOFF, USE OTHER TO BE ALLOWED FOR NUM TYPE COLUMNS, WHERE IF UNEXPANDED NOT ALLOWED
            is_expanded = True

        return is_expanded


    def val_edit(self, col_idx):
        '''Dedicated function for editing QUICK_VAL_DTYPES. Only used as a subfunction for quick_val_dtypes_error_handle().'''
        menu = [f'{v}({k})' for k, v in self.VAL_ALLOWED_DICT.items()]
        allowed = "".join(list(self.VAL_ALLOWED_DICT.keys()))
        while True:
            self.small_preview_object(idx=col_idx)
            _ = (f'Enter new quick validated datatype {", ".join(menu)} > ', allowed)
            _ = self.VAL_ALLOWED_DICT[_]
            if vui.validate_user_str(f'User entered *{_}*, accept? (y/n) > ', 'YN') == 'Y':
                del menu, allowed
                return _


    def quick_val_dtypes_error_handle(self, _QUICK, _GIVEN, kill, error_msg_quip, handling_function_for_given,
                                      fxn, supobj_row_idx=None):
        # ERROR HANDLING FOR MISMATCH OF QUICK_VAL_DTYPES AND GIVEN VALIDATEDDATATYPES ##############################
        supobj_row_idx = self.actv_idx if supobj_row_idx is None else supobj_row_idx

        error_msg = lambda col_idx, _quick, _given: f'\n*** {self.QUICK_HEADER[0][col_idx]} >>> QUICK VALIDATION ' \
                f'DTYPE ({_quick}) DOES NOT MATCH VAL DTYPE PASSED TO {error_msg_quip} ({_given}). ***\n'

        for col_idx in range(self.columns):
            _quick = _QUICK[col_idx]
            _given = _GIVEN[col_idx]
            if _quick != _given:
                if kill: self._exception(fxn, error_msg(col_idx, _quick, _given))
                elif not kill:
                    print(error_msg(col_idx, _quick, _given))
                    while True:
                        _m = vui.validate_user_str(f'\nOVERRIDE QUICK(q) OR GIVEN(g) VALIDATED_DTYPES, OR IGNORE(i) > ', 'QGI')
                        if _m == 'I': break
                        elif _m == 'Q': self.QUICK_VAL_DTYPES[col_idx] = self.val_edit(col_idx)
                        elif _m == 'G': handling_function_for_given(col_idx=col_idx, supobj_row_idx=supobj_row_idx, fxn=fxn)
                        if vui.validate_user_str(f'Continue editing(c) or exit(e) > ', 'CE') == 'E': del _m; break
        del _quick, _given


    def validate_allowed(self, kill=None, fxn=None):
        '''Validate current active index for allowed values.'''

        print(f'\n\033[93m*** BEAR IS VALIDATING ALLOWED ***\033[0m\n')

        # DO NOT OVERWRITE IN CHILD
        fxn = self.get_fxn(fxn, inspect.stack()[0][3])
        kill = akv.arg_kwarg_validater(kill, 'kill', [True, False, None], self.this_module, fxn, return_if_none=True)

        error_msg = lambda _module, column_name, value: \
            f'\n*** COLUMN {column_name} >>> INVALID VALUE *{value}* FOR {_module.upper()}. ALL COLUMNS IN {_module.upper()} ' \
            f'MUST BE IN {self.master_dict()[_module.upper()]["allowed"]} ***\n'

        for _module in self.master_dict():
            SUB_DICT = self.master_dict()[_module]
            _position = SUB_DICT["position"]   # INITIALLY SET _position TO THAT IN MASTER_DICT (STAYS THAT WAY IF supobj_is_full)

            if not self.supobj_is_full:   # IF SUPPORT_OBJECT IS NOT FULL, ONLY VALIDATE FOR ACTIVE MODULE
                if _module.upper() != self.support_name().upper(): continue
                else: _position = self.actv_idx # IF WORKING ON ACTIVE MODULE, SET _position TO self.actv_idx (SHOULD BE ZERO)

            # IF _position IS FILLED W self.empty_value(), SKIP (self.empty_value() WOULD BLOW UP VALIDATION RULES)
            if self.is_empty_getter(_position): continue

            for idx in range(self.columns):
                # MUST CONVERT TO LIST OR CONFUSION BETWEEN np dtypes AND py types (LIKE np.str_ and py.str)
                if len(self.SUPPORT_OBJECT.shape)==1: value = self.SUPPORT_OBJECT.tolist()[idx]
                else: value = self.SUPPORT_OBJECT[_position].tolist()[idx]
                _, __ = SUB_DICT["transform"](value), SUB_DICT["allowed"]

                if not _ in __:
                    ___ = error_msg(_module, self.QUICK_HEADER[0][idx], value)
                    print(___)
                    if kill: self._exception(fxn, ___)
                    elif not kill: self.edit_menu(fxn=fxn, terminate_text=___+f' USER TERMINATED.')
                    del ___

        try: del value, _, __
        except: pass
        del fxn, kill, error_msg, SUB_DICT, _position


    def validate_against_objects(self, OBJECT=None, kill=None, fxn=None):
        # VALIDATE ACCURACY AGAINST DATA OBJECT (IF GIVEN) AND / OR SUPPORT OBJECTS (IF GIVEN)
        # TECHNICALLY ONLY THING THAT CAN BE VALIDATED DIRECTLY AGAINST DATA OBJECT IS VDTYPES.
        # ALL OTHER SUPOBJS CAN ONLY BE VALIDATED AGAINST VDTYPES OR OTHER SUPOBJS.  USE THIS SLOT TO HANDLE BOTH
        # VALIDATION AGAINST OBJECT AND SUPOBJS FOR ValidatedDatatypes, USE IT FOR VALIDATION AGAINST SUPOBJS FOR THE REST.

        print(f'\n\033[93m*** BEAR IS VALIDATING AGAINST OBJECTS ***\033[0m\n')

        fxn = self.get_fxn(fxn, inspect.stack()[0][3])

        kill = akv.arg_kwarg_validater(kill, 'kill', [True, False, None], self.this_module, fxn, return_if_none=False)

        base_error_msg = lambda idx: f'{self.QUICK_HEADER[0][idx]} >>> '
        hdr_error_msg = lambda: f'HEADER CANNOT HAVE ANY VALUES INDICATING "NO VALUE"'
        vdt_error_msg = lambda dtype1, dtype2: f'QUICK VALIDATION OF DATATYPE ({dtype1}) DOES NOT MATCH "OFFICIAL" VALIDATED DATATYPE ({dtype2})'
        mdt_error_msg = lambda idx, moddt, valdt: f'MODIFIED DATATYPE "{moddt}" IS NOT ALLOWED FOR VALIDATED DATATYPE "{valdt}"'
        flt_error_msg = lambda: None
        cut_error_msg = lambda: f'MIN_CUTOFF IS NOT ALLOWED'
        oth_error_msg1 = lambda: f'MINCUTOFFS AND USEOTHER IS NOT ALLOWED ON NUMERICAL COLUMNS'
        oth_error_msg2 = lambda: f'CANNOT APPLY USE_OTHER IF MIN_CUTOFFS IS NOT APPLIED'
        lag_error_msg = lambda lag1, lag2: f'END LAG ({lag2}) MUST BE >= START LAG ({lag1})'
        scl_error_msg = lambda dtype, scale: f'INVALID SCALING VALUE "{scale}" IN MODULE "{self.support_name()}" FOR A COLUMN OF TYPE {dtype}'

        # `IF NO STR DTYPES IN MOD DTYPES, THEN EXPANDED!  ONLY WORKS IF HAVE MOD DTYPES!
        # IF NEVER HAD VAL STR TYPES TO EXPAND, MOD DTYPES COULD HAVE NEVER BEEN A STR TYPE, SO EXPANDED STATE==UNEXPANDED STATE,
        # SO IS EXPANDED. IF HAD VAL STR TYPES, MOD DTYPES HAD STR TYPES, AND IF THEY ARE STILL THERE THEN UNEXPANDED
        # IF HAD VAL STR TYPES, MOD MUST HAVE BEEN STR-TYPES AT SOME POINT, AND IF NO MOD DTYPES ARE STR TYPE, IS EXPANDED
        # VAL DTYPES IS INDETERMINATE BECAUSE THEY DONT CHANGE DURING EXPANSION

        # is_expanded AND self.MOD_DTYPES_HOLDER ONLY NEEDED FOR MINCUTOFFS AS OF 1/15/23
        if not self.mincutoffs_is_empty:    # MAY END UP BEING NEEDED FOR USEOTHER
            is_expanded = self.get_active_mod_dtypes()   # self.MOD_DTYPES_HOLDER UPDATED SILENTLY

        if not self.OBJECT is None and self.qvdtypes_is_empty and \
            (
            not self.vdtypes_is_empty or
            (self.support_name().upper() == 'MODIFIEDDATATYPES' and not self.VDTYPES_is_empty) or
            not self.mdtypes_is_empty or
            (self.support_name().upper() == 'VALIDATEDDATATYPES' and not self.MDTYPES_is_empty) or
            (not self.mincutoffs_is_empty and is_expanded) or
            not self.scaling_is_empty
            ):
            self.get_quick_val_dtypes()


        def error_handle(col_idx, supobj_row_idx, kill, error_msg, fxn):
            if not error_msg is None: print(f'\n*** {error_msg} ***\n')
            if kill: self._exception(fxn, error_msg)
            elif not kill: self.single_edit(col_idx=col_idx, supobj_row_idx=supobj_row_idx, fxn=fxn)
                            # self.edit_menu(fxn=fxn, terminate_text=error_msg)


        # HEADER
        # HEADER CANNOT HAVE ANY VALUES THAT INDICATE EMPTINESS (self.empty_value(), None, '')   "not in"
        # IS THE SAME WHETHER is_expanded AND not is_expanded
        if not self.header_is_empty:
            _position = self.actv_idx if self.support_name()=='HEADER' else self.master_dict()['HEADER']['position']
            for col_idx in range(self.columns):
                if self.SUPPORT_OBJECT[_position][col_idx] in [self.empty_value, '', None]:
                    error_handle(col_idx, _position, kill, base_error_msg(col_idx) + hdr_error_msg(), fxn)
            del _position


        # VDTYPES
        # VALIDATEDDATATYPES CHECKED AGAINST QUICK_VAL_DTYPES
        if not self.OBJECT is None and \
            (not self.vdtypes_is_empty or (self.support_name().upper()=='MODIFIEDDATATYPES' and not self.VDTYPES_is_empty)):

            # IF "REAL" VDTYPES ARE ALREADY IN SUPOBJS, JUST USE THOSE AND CHECK AGAINST QUICK_VAL ######################
            if self.support_name() == 'VALIDATEDDATATYPES':
                if self.vdtypes_is_empty:
                    # IF NO SOURCE OF ValidatedDatatypes IS AVAILABLE WHEN qvdtypes_is_empty, MUST GET qvdtypes FROM Quick (VAL_DTYPES_CHECK)
                    # self.QUICK_VAL_DTYPES STAYS AS CREATED
                    pass
                elif not self.vdtypes_is_empty:
                    for col_idx in range(self.columns):
                        self.quick_val_dtypes_error_handle(self.QUICK_VAL_DTYPES, self.SUPPORT_OBJECT[self.actv_idx],
                                              kill, "ValidatedDatatypes VIA SUPPORT_OBJECT", self.single_edit, fxn)

            elif self.support_name() == 'MODIFIEDDATATYPES':
                # WHEN IN Modified, IF VALIDATED_DATATYPES WAS PASSED VIA KWARG, THIS WILL REPLACE (OR HAS ALREADY
                # REPLACED AND IS EQUAL TO) ANY VAL_DTYPES IN SUPPORT_OBJECT
                _supobj_row_idx = self.master_dict()['VALIDATEDDATATYPES']['position']
                if self.VDTYPES_is_empty and self.vdtypes_is_empty:
                    # IF NO SOURCE OF ValidatedDatatypes IS AVAILABLE WHEN qvdtypes_is_empty, MUST GET qvdtypes FROM Quick (VAL_DTYPES_CHECK)
                    # self.QUICK_VAL_DTYPES STAYS AS CREATED
                    pass
                elif not self.VDTYPES_is_empty:
                    def _handling_function_for_given(col_idx=None, supobj_row_idx=None, fxn=None):
                        _ = self.val_edit(col_idx)
                        self.VALIDATED_DATATYPES[col_idx] = _  # ASSUME THIS HASNT REPLACED VALDTYPES IN SUPOBJ, SO EDIT HERE TOO
                        if not self.vdtypes_is_empty: self.SUPPORT_OBJECT[_supobj_row_idx][col_idx] = _
                        # IF self.vdtypes_is_empty, ASSUME IT WILL BE OVERWRIT LATER W FINALIZED self.VALIDATED_DATATYPES
                        del _

                    self.quick_val_dtypes_error_handle(self.QUICK_VAL_DTYPES, self.VALIDATED_DATATYPES, kill,
                        "ModifiedDatatypes VIA VALIDATED_DATATYPES KWARG", _handling_function_for_given, fxn, supobj_row_idx=_supobj_row_idx)

                    del _handling_function_for_given

                elif not self.vdtypes_is_empty:
                    self.quick_val_dtypes_error_handle(self.QUICK_VAL_DTYPES, self.SUPPORT_OBJECT[_supobj_row_idx], kill,
                          "ModifiedDatatypes VIA SUPPORT_OBJECT", self.single_edit, fxn, supobj_row_idx=_supobj_row_idx)
                del _supobj_row_idx

            elif self.support_name().upper() not in ['VALIDATEDDATATYPES', 'MODIFIEDDATATYPES'] and self.supobj_is_full:
                if not self.vdtypes_is_empty:
                    _supobj_row_idx = self.master_dict()['VALIDATEDDATATYPES']['position']
                    self.quick_val_dtypes_error_handle(self.QUICK_VAL_DTYPES, self.SUPPORT_OBJECT[_supobj_row_idx], kill,
                        f"{self.support_name().upper()} MODULE VIA SUPPORT_OBJECT", self.single_edit, fxn, supobj_row_idx=_supobj_row_idx)
                elif self.vdtypes_is_empty:
                    # IF NO SOURCE OF ValidatedDatatypes IS AVAILABLE WHEN qvdtypes_is_empty, MUST GET qvdtypes FROM QuickValidatedDatatypes
                    # self.QUICK_VAL_DTYPES STAYS AS CREATED
                    pass

            self.qvdtypes_is_empty = False


        # MDTYPES
        # MODIFIEDDATATYPES REVERSE-LOOKUP AGAINST QUICK_VAL_DTYPES (VALIDATEDDATATYPES MAY NOT BE AVAILABLE)
        # REMEMBER THAT ModifiedDatatypes HAS BOTH vdtypes_is_empty and VDTYPES_is_empty
        _GIVEN = None
        if not self.qvdtypes_is_empty:
            if self.support_name().upper()=='VALIDATEDDATATYPES':
                if not self.MDTYPES_is_empty:
                    _GIVEN = self.MODIFIED_DATATYPES
                    error_msg_quip = "ValidatedDatatypes VIA VALIDATED_DATATYPES KWARG"
                elif not self.mdtypes_is_empty:
                    _GIVEN = self.SUPPORT_OBJECT[self.master_dict()['MODIFIEDDATATYPES']['position']]
                    error_msg_quip = "ValidatedDatatypes VIA SUPPORT_OBJECT"
            elif self.support_name().upper() == 'MODIFIEDDATATYPES':
                if not self.mdtypes_is_empty:
                    _GIVEN = self.SUPPORT_OBJECT[self.actv_idx]
                    error_msg_quip = "ModifiedDatatypes VIA SUPPORT_OBJECT"
            elif not self.support_name().upper() in ['VALIDATEDDATATYPES', 'MODIFIEDDATATYPES']:
                if not self.mdtypes_is_empty:
                    _GIVEN = self.SUPPORT_OBJECT[self.master_dict()['MODIFIEDDATATYPES']['position']]
                    error_msg_quip = f"{self.support_name().upper()} VIA SUPPORT_OBJECT"

        if not _GIVEN is None and not self.qvdtypes_is_empty:
            error_msg = lambda col_idx, _quick, _given: f'COLUMN {self.QUICK_HEADER[0][col_idx]} >>> MODIFIED DATATYPE ' \
                f'PASSED TO {error_msg_quip} ({_given}) IS NOT ALLOWED FOR VALIDATED DATATYPE ({_quick})'

            for col_idx in range(self.columns):
                _quick = self.QUICK_VAL_DTYPES[col_idx]
                _given = _GIVEN[col_idx]

                if not _given=='STR' and (self.VAL_REVERSE_LOOKUP[_given] != _quick):     # 4-8-23 ANYTHING CAN BE HANDLED AS A STR
                    print(error_msg(col_idx, _quick, _given))
                    time.sleep(2)
                    if kill: self._exception(fxn, error_msg(col_idx, _quick, _given))
                    if not kill:
                        if self.support_name().upper()=='MODIFIEDDATATYPES':
                            self.single_edit(col_idx=col_idx, fxn=fxn)
                        elif self.support_name().upper not in ['MODIFIEDDATATYPES','VALIDATEDDATATYPES']:
                            self.single_edit(col_idx=col_idx, supobj_row_idx=self.master_dict()['MODIFIEDDATATYPES']['position'], fxn=fxn)
                        elif self.support_name().upper == 'VALIDATEDDATATYPES':
                            mod_posn = self.master_dict()['MODIFIEDDATATYPES']['position']
                            if not self.MDTYPES_is_empty:
                                if _quick in self.VAL_NUM_DTYPES.values():
                                    _ALLOWED, _allowed = self.MOD_NUM_DTYPES.items(), "".join(list(self.MOD_NUM_DTYPES.keys()))
                                elif _quick in self.VAL_TEXT_DTYPES.values():
                                    _ALLOWED, _allowed = self.MOD_TEXT_DTYPES.items(), "".join(list(self.MOD_TEXT_DTYPES.keys()))
                                menu = ", ".join([f'{v}({k})' for k, v in _ALLOWED])
                                while True:
                                    self.small_preview_object(idx=col_idx)
                                    _ = vui.validate_user_str(f'Enter new modified datatype {menu} > ', _allowed)
                                    if vui.validate_user_str(f'User entered *{_}*, accept? (y/n) > ', 'YN') == 'Y':
                                        del menu, _allowed, _ALLOWED
                                        self.MODIFIED_DATATYPES[col_idx] = _
                                        if not self.mdtypes_is_empty: self.SUPPORT_OBJECT[mod_posn][col_idx] = _
                            elif not self.mdtypes_is_empty:
                                self.single_edit(col_idx=col_idx, supobj_row_idx=mod_posn, fxn=fxn)
                            del mod_posn
            del error_msg


        # FILTERING
        # FILTERING CAN APPLY TO ANY COLUMN, BEFORE AND AFTER EXPANSION


        # FIND COLUMNS THAT ARE ALLOWED TO HAVE MINCUTOFF AND USEOTHER (VARIES BASED ON is_expand /


        if not self.mincutoffs_is_empty or not self.useother_is_empty:
            CUTOFF_ALLOWED = np.fromiter((False for _ in range(self.columns)), dtype=bool)
            if not is_expanded and not np.array_equiv(self.MOD_DTYPES_HOLDER,
                                      np.fromiter((self.empty_value for _ in range(self.columns)), dtype=object)):
                # MUST GET is_empty/not is_empty STATE FOR MDTYPES_HOLDER THE HARD WAY BECAUSE CANT PASS AN OBJ TO self.is_empty_getter()
                for col_idx in range(self.columns):
                    if self.MOD_DTYPES_HOLDER[col_idx] in self.MOD_TEXT_DTYPES.values():
                        CUTOFF_ALLOWED[col_idx] = True
            elif is_expanded and not self.qvdtypes_is_empty:
                for col_idx in range(self.columns):
                    _type = self.QUICK_VAL_DTYPES[col_idx]
                    if _type in self.VAL_TEXT_DTYPES.values() or (_type == 'INT' and self.MOD_DTYPES_HOLDER[col_idx]=='STR'):
                        CUTOFF_ALLOWED[col_idx] = True
                del _type, is_expanded


        # MINCUTOFFS
        ''' 1/2/23 
        # PRE-EXPAND:
        # MIN_CUTOFF ONLY SENSIBLY APPLIES TO STR-TYPE M_DTYPES (INT VALTYPE COULD BE MADE INTO MODTYPE STR)
        # MEANING THIS COULD BE APPLIED TO STR M_DTYPES 'STR', 'SPLIT_STR', 'NNLM50', WHICH BECOME BIN, INT OR FLOAT
        # HAVE TO LOOK IN PreRunFilter TO SEE WHAT HAPPENS TO MODTYPE FLOAT/BIN/INT COLUMNS IF A NON-ZERO NUMBER IS HERE
        # (THINKING IT JUST BYPASSES BASED ON TYPE, W/O EVEN LOOKING AT NUMBER).  SO ONLY STR-MODTYPES CAN HAVE MINCUTOFF
        # AFTER EXPAND:
        # MINCUTOFF COULD APPLY TO MTYPE 'BIN' (VIA 'STR' AND 'SPLIT_STR') 'INT' (VIA 'SPLIT_STR') OR 
        # 'FLOAT' (VIA NNLM50). A COLUMN OF ANY MTYPE COULD HOLD A NON-ZERO MIN CUTOFF, BECAUSE THINGS 
        # THAT WERE "STR" CAN BECOME ANY NUMBER DTYPE WHEN EXPANDED. VTYPE BIN AND FLOAT CANNOT HAVE CUTOFF.'''

        if not self.mincutoffs_is_empty:
            _position = self.master_dict()['MINCUTOFFS']['position'] if self.support_name().upper() != 'MINCUTOFFS' \
                else self.actv_idx
            for col_idx in range(self.columns):
                value = self.SUPPORT_OBJECT[_position][col_idx]
                if value > 0 and CUTOFF_ALLOWED[col_idx] is False:
                    error_msg = base_error_msg(col_idx)+cut_error_msg()
                    error_handle(col_idx, _position, kill, error_msg, fxn)
                    del error_msg
            del _position, value


        # USEOTHER
        # USEOTHER CAN ONLY BE 'Y' IF MINCUTOFF > 0
        if not self.useother_is_empty:
            if self.mincutoffs_is_empty:
                _position = self.master_dict()['USEOTHER']['position'] if not self.support_name() == 'USEOTHER' else self.actv_idx
                _other = self.SUPPORT_OBJECT[_position]
                for col_idx in range(self.columns):
                    if _other[col_idx] == 'Y' and CUTOFF_ALLOWED[col_idx] is False:
                        error_handle(col_idx, _position, kill, base_error_msg(col_idx) + oth_error_msg1(), fxn)
                del _position, _other
            elif not self.mincutoffs_is_empty:
                # A FULL SUPPORT_OBJECT MUST HAVE BEEN PASSED, SO DOESNT MATTER WHAT self.actv_idx IS
                # (THIS WOULD CHANGE IF EVER ALLOW PASS OF USE_OTHER TO MINCUTOFFS AND VICE VERSA)
                _cutoff = self.SUPPORT_OBJECT[self.master_dict()['MINCUTOFFS']['position']]
                _position = self.master_dict()['USEOTHER']['position'] #if not self.support_name() == 'USEOTHER' else self.actv_idx
                _other = self.SUPPORT_OBJECT[_position]
                for col_idx in range(self.columns):
                    if _other[col_idx] == 'Y' and _cutoff[col_idx] == 0:
                        error_handle(col_idx, _position, kill, base_error_msg(col_idx) + oth_error_msg2(), fxn)
                del _position, _other, _cutoff

        # STARTLAG, ENDLAG
        # ANYTHING CAN BE LAGGED PRE-EXPANSION, SO POST-EXPANSION COULD BE LAG FOR ALL MOD DTYPES
        # START LAG MUST BE <= END_LAG, END LAG MUST BE >= START_LAG
        if not self.startlag_is_empty and not self.endlag_is_empty:
            # A FULL SUPPORT_OBJECT MUST HAVE BEEN PASSED, SO DOESNT MATTER WHAT self.actv_idx IS
            _position1 = self.master_dict()['STARTLAG']['position'] # if not self.support_name().upper()=='STARTLAG' else self.actv_idx
            _position2 = self.master_dict()['ENDLAG']['position'] # if not self.support_name().upper()=='ENDLAG' else self.actv_idx
            error_msg = lambda col_idx, lag1, lag2: base_error_msg(col_idx) + lag_error_msg(lag1, lag2)
            for col_idx in range(self.columns):
                lag1 = self.SUPPORT_OBJECT[_position1][col_idx]
                lag2 = self.SUPPORT_OBJECT[_position2][col_idx]
                if not lag2 >= lag1:
                    if kill: self._exception(fxn, error_msg(col_idx, lag1, lag2))
                    elif not kill:
                        if vui.validate_user_str(f'Change start lag(s) or end lag(e) > ', 'SE') == 'S':
                            error_handle(col_idx, _position1, kill, error_msg(col_idx, lag1, lag2), fxn)
                        else: error_handle(col_idx, _position2, kill, error_msg(col_idx, lag1, lag2), fxn)
            del _position1, _position2, lag1, lag2

        # SCALING
        # SCALING CAN ONLY APPLY TO VAL DTYPE "INT" AND "FLOAT" COLUMNS, BEFORE AND AFTER EXPANSION
        # WOULDNT HAVE TO SCALE BIN, ONLY OTHER INT AND FLOAT SHOW UP AFTER SPLIT_STR AND NNLM50, UNLIKELY TO NEED SCALING
        if not self.scaling_is_empty:    # THIS TRIGGERED TO get_quick_vdtypes() ABOVE
            _position = self.master_dict()['SCALING']['position'] if not self.support_name().upper()=='SCALING' else self.actv_idx
            for col_idx in range(self.columns):
                _ = self.QUICK_VAL_DTYPES[col_idx]
                __ = self.SUPPORT_OBJECT[_position][col_idx]
                if _ not in ['INT', 'FLOAT'] and __ != '':
                    error_handle(col_idx, _position, kill, base_error_msg(col_idx)+scl_error_msg(_, __), fxn)
            del _position, _, __

        del kill, fxn, error_handle, base_error_msg, hdr_error_msg, vdt_error_msg, mdt_error_msg, \
            flt_error_msg, cut_error_msg, oth_error_msg1, oth_error_msg2, lag_error_msg, scl_error_msg



    def value_validation(self, value, supobj_name, kill=None, fxn=None):
        kill = akv.arg_kwarg_validater(kill, 'kill', [True, False, None], self.this_module, fxn, return_if_none=True)

        fxn = self.get_fxn(fxn, inspect.stack()[0][3])

        if value not in self.master_dict()[supobj_name.upper()]['allowed']:
            error_msg = f'ILLEGAL VALUE "{value}" PASSED TO value IN {fxn}. ' \
                        f'{self.support_name()} MUST BE IN {self.master_dict()[supobj_name.upper()]["allowed"]}.'
            if kill:
                self._exception(fxn, error_msg)

            elif not kill:
                print(f'\n*** {error_msg} ***\n')
                __ = vui.validate_user_str(f'Manual override(m), select from ALLOWED(s), terminate(t) > ', 'MST')
                if __ == 'T': self._exception(fxn, error_msg)
                elif __ == 'M':
                    while True:
                        value = input(f'Enter new value (case sensitive) > ')
                        if vui.validate_user_str(f'User entered "{value}", accept? (y/n) > ', 'YN'):
                            break
                elif __ =='S':
                    value = self.ALLOWED_LIST[ls.list_single_select(self.ALLOWED_LIST, f'Select new value', 'idx')[0]]

                del error_msg, __, kill

                return value


    def preview_object(self, _rows_, _columns_):

        if self.OBJECT is None:
            print(f'\n *** OBJECT WAS NOT GIVEN, CANNOT PRINT *** \n')
        else:
            _start_col = ls.list_single_select(self.QUICK_HEADER[0], f'Select start column > ', 'idx')[0]
            _end_col = ls.list_single_select(self.QUICK_HEADER[0][_start_col:], f'Select end column > ', 'idx')[0] + _start_col
            pop.print_object_preview(
                                    self.OBJECT,
                                    'OBJECT',
                                    _rows_,     # GOES TO ioap, ROWS TO PRINT NOT ROWS IN OBJECT!
                                    _columns_,  # COLUMNS TO PRINT NOT COLUMNS IN DATA!
                                    _start_col,
                                    _end_col,
                                    orientation='column',
                                    header=''
                                    )
            del _start_col, _end_col


    def small_preview_object(self, idx=None):
        # IF idx IS GIVEN, ONLY PRINTS THAT COLUMN; IF NOT GIVEN, PRINTS ALL
        sopfa(self.OBJECT, self.object_given_orientation,
                SINGLE_OR_FULL_SUPPORT_OBJECT=self.SUPPORT_OBJECT, support_name=self.support_name(), idx=idx)


    def view_support_objects(self):

        fxn = inspect.stack()[0][3]

        if self.supobj_is_full:
            psc.PrintSupportObjects_NewObjects(
                                                 self.OBJECT,
                                                 self.support_name(),
                                                 self.SUPPORT_OBJECT,
                                                 orientation=self.object_given_orientation,
                                                 _columns=self.columns,
                                                 max_hdr_len=max(list(map(len, self.SUPPORT_OBJECT[0]))),
                                                 calling_module=self.calling_module,
                                                 calling_fxn=self.calling_fxn
                                                 )

        elif not self.supobj_is_full:
            psc.PrintSingleSupportObject(
                                                 self.SUPPORT_OBJECT[self.actv_idx],
                                                 self.support_name(),
                                                 self.OBJECT_HEADER,
                                                 calling_module=self.calling_module,
                                                 calling_fxn=fxn
                                                 )


    def edit_menu(self, fxn=None, allowed='EMFRASIDBCXT', terminate_text=None):

        fxn = self.get_fxn(fxn, inspect.stack()[0][3])

        allowed_str = 'EMFRASIDBCXT'

        [self._exception(inspect.stack()[0][3], f'INVALID CHAR IN allowed') for char in allowed if char not in allowed_str]

        ALLOWED_MENU = [f'Single edit(e)',
                        f'Multi edit(m)',
                        f'Full edit(f)',
                        f'Restore defaults(r)',
                        f'Manual fill all(a)',
                        f'Select from allowed(s)',
                        f'Ignore/abort(i)',
                        f'print DATA OBJECT(d)',
                        f'print SUPPORT OBJECT(b)',
                        f'reset and restart(c)',
                        f'accept and exit(x)',
                        f'Terminate(t)'
                        ]

        SUPPORT_OBJECT_BACKUP = self.SUPPORT_OBJECT.copy()

        menu_str = "\n"
        for i, v in enumerate(allowed_str):
            if v in allowed: menu_str += f'{ALLOWED_MENU[i]}'.ljust(40)
            if (i+1) % 3 == 0: menu_str += f"\n"

        while True:
            __ = vui.validate_user_str(menu_str + f'\n > ', allowed)
            if __ == 'E': self.single_edit(fxn=fxn)
            elif __ == 'M': self.multi_edit()
            elif __ == 'F': self.full_edit(fxn=fxn)
            elif __ == 'R': self.default_fill()
            elif __ == 'A': self.manual_fill_all(value=input(f'Enter value for all columns (case sensitive) > '),
                                                 supobj_name=self.support_name().upper(), fxn=fxn)
            elif __ == 'S': self.single_select_from_allowed()
            elif __ == 'I': break
            elif __ == 'D': self.preview_object(20, 10)
            elif __ == 'B': self.view_support_objects()
            elif __ == 'C': self.SUPPORT_OBJECT = SUPPORT_OBJECT_BACKUP.copy()
            elif __ == 'X': break
            elif __ == 'T': self._exception(fxn, terminate_text if not terminate_text is None else f'USER TERMINATED IN edit_menu()')

        del SUPPORT_OBJECT_BACKUP, ALLOWED_MENU, menu_str

        if not self.bypass_validation:
            self.validate_allowed(kill=not self.prompt_to_override, fxn=fxn)
            self.validate_against_objects(kill=not self.prompt_to_override, fxn=fxn)


    def single_edit(self, col_idx=None, supobj_row_idx=None, fxn=None):
        '''Prompt to initiate edits to individual entries.'''
        # SHOULDNT BE ABLE TO GET HERE W Filtering & Scaling, override IS NOT ALLOWED
        fxn = self.get_fxn(fxn, inspect.stack()[0][3])

        supobj_row_idx = akv.arg_kwarg_validater(supobj_row_idx, 'sup_obj_as_row_idx', list(range(len(self.master_dict())))+[None],
                         self.this_module, fxn, return_if_none=self.actv_idx)

        reverse_dict = dict(((self.master_dict()[_]['position'], _) for _ in self.master_dict()))
        supobj_name = reverse_dict[supobj_row_idx].upper()

        del reverse_dict, supobj_row_idx

        # IF col_idx IS NOT GIVEN, ALLOW TO CHOOSE
        col_idx = akv.arg_kwarg_validater(col_idx, 'col_idx', list(range(self.columns))+[None], self.this_module, fxn)
        if col_idx is None:
            col_idx = ls.list_single_select(self.QUICK_HEADER[0], f'Select column to edit {supobj_name}', 'idx')[0]

        self.small_preview_object(col_idx)

        if supobj_name.upper() == 'HEADER':
            while True:
                new_value = input(f'ENTER NEW COLUMN NAME FOR {self.QUICK_HEADER[0][col_idx]} > ')
                if vui.validate_user_str(f'USER ENTERED "{new_value}", accept? (y/n) > ', 'YN') == 'Y':
                    if new_value in np.delete(self.SUPPORT_OBJECT[self.actv_idx].copy(), col_idx, axis=0):
                        print(f'\n*** NEW COLUMN NAME IS ALREADY IN HEADER, CANNOT HAVE DUPLICATE COLUMN NAMES ***\n')
                        continue
                    else: break

        elif supobj_name in ['VALIDATEDDATATYPES', 'MODIFIEDDATATYPES', 'USEOTHER']:
            TEMP_ALLOWED_DICT = {
                'VALIDATEDDATATYPES': self.VAL_ALLOWED_DICT,
                'MODIFIEDDATATYPES': self.MOD_ALLOWED_DICT,
                'USEOTHER': {'Y':'Y', 'N':'N'}
            }
            _ALLOWED = ", ".join([(f'{v}({k})') for k,v in TEMP_ALLOWED_DICT[supobj_name.upper()].items()])
            _allowed = "".join(TEMP_ALLOWED_DICT[supobj_name.upper()].keys()).upper()
            while True:
                new_value = vui.validate_user_str(f'ENTER NEW {supobj_name} FOR COLUMN {self.QUICK_HEADER[0][col_idx]} --- {_ALLOWED} > ',
                                                  _allowed)
                if vui.validate_user_str(f'USER ENTERED "{TEMP_ALLOWED_DICT[supobj_name.upper()][new_value]}", accept? (y/n) > ', 'YN') == 'Y':
                    new_value = self.value_validation(new_value, supobj_name.upper(), kill=not self.prompt_to_override, fxn=fxn)
                    del TEMP_ALLOWED_DICT, _ALLOWED, _allowed; break

        elif supobj_name.upper() in ['MINCUTOFFS', 'STARTLAG', 'ENDLAG']:
            _ = self.master_dict()[supobj_name]["allowed"]
            while True:
                new_value = vui.validate_user_int(f'ENTER NEW {supobj_name.upper()} FOR COLUMN {self.QUICK_HEADER[0][col_idx]} --- '
                      f'{_} > ', min=min(_), max=max(_))

                if vui.validate_user_str(f'USER ENTERED "{new_value}", accept? (y/n) > ', 'YN') == 'Y':
                    new_value = self.value_validation(new_value, supobj_name.upper(), kill=not self.prompt_to_override, fxn=fxn)
                    del _; break

        del supobj_name

        return new_value


    def multi_edit(self):
        '''Display all values and prompt for edit of many entries.'''

        fxn = inspect.stack()[0][3]

        while True:
            # self.small_preview_object()
            # self.view_support_objects()
            while True:
                edit_idx = ls.list_single_select(self.SUPPORT_OBJECT[0], f'Select column to override', 'idx')
                self.SUPPORT_OBJECT[self.actv_idx][edit_idx] = self.single_edit(col_idx=edit_idx, fxn=fxn)
                while True:
                    __ =  vui.validate_user_str(f'\nEdit another(y), end(e), print current state(p) > ', 'YEP')
                    if __ == 'P': self.small_preview_object(); continue
                    elif __ in 'YE': break
                if __ == 'E': del edit_idx, __; break
                elif __ == 'Y': continue

            # self.small_preview_object()
            # self.view_support_objects()
            if vui.validate_user_str(f'Accept? (y/n) > ', 'YN') == 'Y': break


    def full_edit(self, fxn=None):
        fxn = self.get_fxn(fxn, inspect.stack()[0][3])

        # self.small_preview_object()
        # self.view_support_objects()
        for col_idx in range(self.columns):
            self.small_preview_object(idx=col_idx)
            print()
            self.SUPPORT_OBJECT[self.actv_idx] = self.single_edit(col_idx=col_idx, fxn=fxn)


    def single_select_from_allowed(self):
        return ls.list_single_select(self.ALLOWED_LIST, f'Select from allowed values', 'value')[0]


    def manual_fill_all(self, value, supobj_name, fxn=None):
        fxn = fxn if not fxn in None else inspect.stack()[0][3]
        if not self.bypass_validation:
            value = self.value_validation(value, supobj_name.upper(), kill=not self.prompt_to_override, fxn=fxn)

        self.SUPPORT_OBJECT[self.actv_idx] = np.fromiter((value for _ in range(self.columns)), dtype=object)


    def fill_all_from_allowed(self):
        new_value = self.single_select_from_allowed()
        self.SUPPORT_OBJECT[self.actv_idx] = np.fromiter((new_value for idx in range(self.columns)), dtype=object)
        del new_value


    def default_fill(self):
        # OVERWRITE IN CHILD
        self.SUPPORT_OBJECT[self.actv_idx] = np.fromiter((self.default_value() for _ in range(self.columns)), dtype=object)


    def fill_from_kwarg(self):
        # OVERWRITE IN CHILD
        pass


    def delete_column(self, idx):
        '''Delete column support values at given idx.'''
        #    BEAR MAYBE THIS SHOULD BE LEFT TO A HIGHER FUNCTION THAT BINDS ALL NINE CHILD CLASSES
        self.SUPPORT_OBJECT = np.delete(self.SUPPORT_OBJECT, idx, axis=1)


    def insert_idx(self, idx, VALUES_AS_LIST):
        '''Insert value at given idx.'''
        # if not self.bypass_validation:
        #    BEAR THIS ONE IS GOING TO BE HARD, TO VALIDATE VALUES FOR 9 TYPES OF SUPPORT OBJECTS
        #    MAYBE THIS SHOULD BE LEFT TO A HIGHER FUNCTION THAT BINDS ALL NINE CHILD CLASSES
        #     value = self.value_validation(value)

        self.SUPPORT_OBJECT = np.insert(self.SUPPORT_OBJECT, idx, VALUES_AS_LIST, axis=1)



















if __name__ == '__main__':

    # TEST MODULE
    # AS OF 1/4/23, THE INTENT IS TO DO SOME TESTING HERE TO GET THIS AND THE CHILDREN GOING,
    # BUT MAJORITY OF TEST DONE IN BuildFullSupportObject

    pass





























































































































































