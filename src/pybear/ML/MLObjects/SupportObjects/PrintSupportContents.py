import numpy as np
import sys
from data_validation import arg_kwarg_validater as akv
from ML_PACKAGE._data_validation import list_dict_validater as ldv
from debug import get_module_name as gmn
from general_data_ops import get_shape as gs
from MLObjects.SupportObjects import master_support_object_dict as msod


# ApexPrintSupportObjects_OldObjects, PrintSupportObjects_NewObjects, PrintSupportObjects_OldObjects
# PRINT LIKE THIS:

'''
****************************************************************************************************
DATA HYPERPARAMETER SETUP :
                                                                       S                                       
                                                  C                    C        F                              
                                    D       D     U     O   S          A        I                              
                                    T       T     T     T   T          L        L                              
                                  V Y     M Y   M O   U H   A L   E L  I        T                              
COL                               A P     O P   I F   S E   R A   N A  N        E                              
IDX  COLUMN                       L E     D E   N F   E R   T G   D G  G        R                              
------------------------------------------------------------------------------------------------------------------------
1    RUMPELSTILTSKIN COLUMN1   NNLM50  NNLM50  None     N  None     0  these    steak                          
                                                                                tater
                                                                                gravy
2    RUMPELSTILTSKIN COLUMN2   NNLM50  NNLM50  None     N  None     0  are      steak                          
                                                                                tater
                                                                                gravy
3    RUMPELSTILTSKIN COLUMN3   NNLM50  NNLM50  None     N  None     0  the      steak                          
                                                                                tater
                                                                                gravy
4    RUMPELSTILTSKIN COLUMN4   NNLM50  NNLM50  None     N  None     0  times    steak                          
                                                                                tater
                                                                                gravy
5    RUMPELSTILTSKIN COLUMN5   NNLM50  NNLM50  None     N  None     0  that     steak                          
                                                                                tater
                                                                                gravy

****************************************************************************************************
'''


class ApexPrintSupportObjects:
    '''12/31/22 Parent for SupportObjectPrint_OldObjects, SupportObjectPrint_NewObjects, SingleSupportObjectPrint.
       Once (if) PreRunFilter is converted to (New) single support object format (and any other objects that may still need
       the old way if has individual support objects) the SupportObjectPrint_OldObjects class can be deleted.'''

    # prints column name, validated type, user type, min cutoff, use other, start lag, end lag, scaling &
    # filtering for all columns

    def __init__(self,
                 OBJECT,
                 name,
                 SUPPORT_OBJECT_HOLDER,
                 orientation=None,
                 _columns=None,  # FOR SPEED, IF ALREADY AVAILABLE FROM CALLER
                 max_hdr_len=None,
                 calling_module=None,
                 calling_fxn=None
                 ):

        self.calling_module = calling_module if not calling_module is None else gmn.get_module_name(str(sys.modules[__name__]))
        self.calling_fxn = calling_fxn if not calling_fxn is None else '__init__'

        OBJECT = ldv.list_dict_validater(OBJECT, name)[1]
        orientation = akv.arg_kwarg_validater(orientation, 'orientation', ['COLUMN', 'ROW'], self.calling_module,
                                                   self.calling_fxn)

        if _columns is None:
            if orientation is None:
                self._exception(f'if orientation is not given, _columns must be given, and vice versa.')

            _columns = gs.get_shape(name, OBJECT, orientation)[1]


        if _columns != len(SUPPORT_OBJECT_HOLDER[0]):
            self._exception(f'NUMBER OF COLUMNS IN HEADER MUST MATCH LENGTH OF SUPPORT OBJECTS')


        # GET MAX_HDR_LEN OF ALL OBJECTS, FOR SYMMETRY OF ALL OBJECTS IN PRINTOUT
        for supobj_idx in range(9):
            try:
                all_max_hdr_len = max_hdr_len if not max_hdr_len is None else max(list(map(len, SUPPORT_OBJECT_HOLDER[supobj_idx])))
                break
            except: continue

        gap = f'  '

        max_col_char = 30 + 20 * int(max(map(len, SUPPORT_OBJECT_HOLDER[msod.QUICK_POSN_DICT()["FILTERING"]]))==0) # IF NO FILTERING, ALLOW COLUMN NAME TO BE BIGGER
        index_width = max(list(map(len, list(map(str, range(_columns)))))) + 4
        # ALLOW UP TO max_col_char CHARS FOR COLUMN NAME, BUT AT LEAST 8 FOR 'COLUMN  '
        column_width = max(min(all_max_hdr_len + 2, max_col_char), len('COLUMN') + 2)
        vdtype_width = 7
        mdtype_width = 10
        min_cutoff_width = 6
        use_other_width = 6
        slag_width = 6
        elag_width = 6
        scaling_width = int(max(map(len, SUPPORT_OBJECT_HOLDER[8]))) + 2
        filtering_width = 6 + (50-column_width) * int(max(map(len, SUPPORT_OBJECT_HOLDER[3]))!=0)   # IF NO FILTERING, MAKE FILTERING SHORTER
        gap_width = len(gap)
        total_len_to_filt = np.sum((index_width, column_width, vdtype_width, mdtype_width, min_cutoff_width, use_other_width,
                                    slag_width, elag_width, scaling_width, gap_width*2))

        def row_print_template(index, column, vdtype, mdtype, min_cutoff, use_other, slag, elag, scaling, filtering):
            print(str(index).ljust(index_width) + column[:max_col_char].ljust(column_width) + vdtype.rjust(vdtype_width) +
                mdtype.rjust(mdtype_width) + str(min_cutoff).rjust(min_cutoff_width) + use_other.rjust(use_other_width) +
                str(slag).rjust(slag_width) + str(elag).rjust(elag_width) + gap + scaling.ljust(scaling_width) +
                gap + filtering.ljust(filtering_width))


        print(f'\n{name} HYPERPARAMETER SETUP :')

        row_print_template('    ', '    ', '    ', '    ', '    ', '    ', '    ', '    ', 'S   ', '    ')
        row_print_template('    ', '    ', '    ', '    ', '   C', '    ', '    ', '    ', 'C   ', 'F   ')
        row_print_template('    ', '    ', '   D', '   D', '   U', '   O', ' S  ', '    ', 'A   ', 'I   ')
        row_print_template('    ', '    ', '   T', '   T', '   T', '   T', ' T  ', '    ', 'L   ', 'L   ')
        row_print_template('    ', '    ', ' V Y', ' M Y', ' M O', ' U H', ' A L', ' E L', 'I   ', 'T   ')
        row_print_template('COL ', '    ', ' A P', ' O P', ' I F', ' S E', ' R A', ' N A', 'N   ', 'E   ')
        row_print_template('IDX ','COLUMN',' L E', ' D E', ' N F', ' E R', ' T G', ' D G', 'G   ', 'R   ')
        print(f'-'*120)
        for idx in range(len(SUPPORT_OBJECT_HOLDER[0])):
            if len(SUPPORT_OBJECT_HOLDER[3][idx])==0:
                row_print_template(str(idx+1), *SUPPORT_OBJECT_HOLDER[[0,1,2,4,5,6,7,8],idx], '')
            else:
                row_print_template(str(idx+1), *SUPPORT_OBJECT_HOLDER[[0,1,2,4,5,6,7,8],idx], SUPPORT_OBJECT_HOLDER[3][idx][0][:filtering_width])
                for filter_step in SUPPORT_OBJECT_HOLDER[3][idx][1:][:filtering_width]:
                    print(' '*total_len_to_filt + filter_step)

        del row_print_template

        del gap, max_col_char, index_width, column_width, vdtype_width, mdtype_width, min_cutoff_width, use_other_width, \
            slag_width, elag_width, scaling_width, filtering_width, gap_width, total_len_to_filt


    def _exception(self,  error_msg):
        raise Exception(f'\n*** {self.calling_module}.{self.calling_fxn}() >>> {error_msg}')




class PrintSupportObjects_NewObjects(ApexPrintSupportObjects):
    '''Prints all support objects ingested as a single FULL_SUPOBJ.'''

    # prints column name, validated type, user type, min cutoff, "other", & filtering for all columns in all objects

    def __init__(self,
                 OBJECT,
                 name,
                 SUPPORT_OBJECT_HOLDER,
                 orientation=None,
                 _columns=None,  # FOR SPEED, IF ALREADY AVAILABLE FROM CALLER
                 max_hdr_len=None,
                 calling_module=None,
                 calling_fxn=None
                 ):

        OBJECT = ldv.list_dict_validater(OBJECT, name)[1]

        self.calling_module = calling_module if not calling_module is None else gmn.get_module_name(str(sys.modules[__name__]))
        self.calling_fxn = calling_fxn if not calling_fxn is None else '__init__'

        orientation = akv.arg_kwarg_validater(orientation, 'orientation', ['COLUMN', 'ROW'], self.calling_module, self.calling_fxn)

        super().__init__(OBJECT, name, SUPPORT_OBJECT_HOLDER,orientation=orientation, _columns=_columns,
                max_hdr_len=max_hdr_len, calling_module=self.calling_module, calling_fxn=self.calling_fxn)


    # INHERITS
    # _exception





class PrintSupportObjects_OldObjects(ApexPrintSupportObjects):
    '''Prints all support objects ingested as indiviual support objects.'''
    def __init__(self,
                 OBJECT,      # SELECT OBJECT OUT OF SRNL OR SWNL AT INSTANTIATION
                 name,
                 orientation=None,
                 _columns=None,
                 HEADER=None,
                 VALIDATED_DATATYPES=None,
                 MODIFIED_DATATYPES=None,
                 FILTERING=None,
                 MIN_CUTOFFS=None,
                 USE_OTHER=None,
                 START_LAG=None,
                 END_LAG=None,
                 SCALING=None,
                 max_hdr_len=None,
                 calling_module=None,
                 calling_fxn=None
                 ):

        OBJECT = ldv.list_dict_validater(OBJECT, name)[1]

        self.calling_module = calling_module if not calling_module is None else gmn.get_module_name(str(sys.modules[__name__]))
        self.calling_fxn = calling_fxn if not calling_fxn is None else '__init__'

        orientation = akv.arg_kwarg_validater(orientation, 'orientation', ['COLUMN', 'ROW'], self.calling_module, self.calling_fxn)

        SUPOBJ_NAMES = ("HEADER", "VALIDATEDDATATYPES", "MODIFIEDDATATYPES", "FILTERING", "MINCUTOFFS", "USEOTHER", "STARTLAG",
                      "ENDLAG", "SCALING")
        OBJ_TUPLE = (HEADER, VALIDATED_DATATYPES, MODIFIED_DATATYPES, FILTERING, MIN_CUTOFFS,
                            USE_OTHER, START_LAG, END_LAG, SCALING)

        if _columns is None:
            if orientation is None:
                self._exception(f'if orientation is not given, _columns must be given, and vice versa.')

            _columns = gs.get_shape(name, OBJECT, orientation)[1]

        # BUILD THE 2023 FORMAT OF SUPPORT_OBJECTS!
        SUPPORT_OBJECT_HOLDER = msod.build_empty_support_object(_columns)
        for supobj_name, SUPPORT_OBJECT in zip(SUPOBJ_NAMES, OBJ_TUPLE):
            if not SUPPORT_OBJECT is None:
                SUPPORT_OBJECT = ldv.list_dict_validater(SUPPORT_OBJECT, name)[1][0]
                if len(SUPPORT_OBJECT) != _columns:
                    self._exception(f'SUPPORT OBJECTS ({supobj_name}) MUST HAVE LEN EQUAL TO NUMBER OF COLUMNS IN OBJECT')
                SUPPORT_OBJECT_HOLDER[msod.QUICK_POSN_DICT()[supobj_name]] = SUPPORT_OBJECT

        super().__init__(OBJECT, name, SUPPORT_OBJECT_HOLDER, orientation=orientation, _columns=_columns,
                max_hdr_len=max_hdr_len, calling_module=self.calling_module, calling_fxn=self.calling_fxn)


    # INHERITS
    # _exception















# PrintSingleSupportObject
# PRINTS LIKE:
'''
****************************************************************************************************

VALIDATEDDATATYPES SETUP :
IDX  COLUMN                          VALIDATEDDATATYPES                  
------------------------------------------------------------------------------------------------------------------------
1)   RUMPELSTILTSKIN COLUMN1         NNLM50
2)   RUMPELSTILTSKIN COLUMN2         NNLM50
3)   RUMPELSTILTSKIN COLUMN3         NNLM50
4)   RUMPELSTILTSKIN COLUMN4         NNLM50
5)   RUMPELSTILTSKIN COLUMN5         NNLM50

****************************************************************************************************
'''





class PrintSingleSupportObject(ApexPrintSupportObjects):   # ONLY USING _exception
    '''Prints a single support object.'''
    # prints individual support object

    def __init__(self,
                 SUPPORT_OBJECT,
                 name,
                 HEADER=None,
                 calling_module=None,
                 calling_fxn=None
                 ):

        self.calling_module = calling_module if not calling_module is None else gmn.get_module_name(str(sys.modules[__name__]))
        self.calling_fxn = calling_fxn if not calling_fxn is None else '__init__'

        # DONT DO ldv ON SUPPORT_OBJECT, FILTERING WOULD GET RUINED
        if not isinstance(SUPPORT_OBJECT, (np.ndarray, list, tuple)):
            raise Exception(f'{name} MUST BE A LIST-TYPE')

        name = name.upper() if isinstance(name, str) else self._exception(f'"name" must be a string')

        ALLOWED_NAMES = ['HEADER', 'VALIDATEDDATATYPES', 'MODIFIEDDATATYPES', 'FILTERING', 'MINCUTOFFS',
                                    'USEOTHER', 'STARTLAG', 'ENDLAG', 'SCALING']
        if name not in ALLOWED_NAMES:
            self._exception(f'ILLEGAL OBJECT NAME "{name}", MUST BE IN {", ".join(ALLOWED_NAMES)}')

        if name != 'FILTERING': SUPPORT_OBJECT = SUPPORT_OBJECT.reshape((1,-1))[0]
        # JUST KEEP FILTERING AS IS
        else: SUPPORT_OBJECT = SUPPORT_OBJECT

        if HEADER is None:
            HEADER = np.fromiter((f'COLUMN{_+1}' for _ in range(len(self.SUPPORT_OBJECT))), dtype='<U15').reshape((1,-1))
        else:
            header_type, HEADER = ldv.list_dict_validater(HEADER, 'HEADER')
            if not header_type == 'ARRAY': raise Exception(f'HEADER MUST BE A LIST-TYPE THAN CAN BE CONVERTED TO AN NP ARRAY')
            del header_type

        # GET MAX_HDR_LEN OF ALL COLUMNS, FOR SYMMETRY OF ALL OBJECTS IN PRINTOUT
        max_hdr_len = max(list(map(len, HEADER[0]))) + 6

        index_width = max(list(map(len, list(map(str, range(len(SUPPORT_OBJECT))))))) + 4
        column_width = min(max_hdr_len + 2, 60)
        len_to_object = index_width + column_width + 1

        print(f'\n{name} SETUP :')
        print(f'IDX'.ljust(index_width) + [f'COLUMN'.ljust(column_width) if not name == 'HEADER' else ''][0] +
              f' {name.upper()}'.ljust(len_to_object))
        print(f'-' * 120)
        template = lambda idx, column, value: print(f'{idx})'.ljust(index_width) + f'{column}'.ljust(column_width), f'{value}')
        if name.upper()=='FILTERING':
            for idx in range(len(SUPPORT_OBJECT)):
                if len(SUPPORT_OBJECT[idx])==0:
                    template(str(idx+1), HEADER[0][idx], [])
                else:
                    template(str(idx+1), HEADER[0][idx], SUPPORT_OBJECT[idx][0])
                    for filter_step in SUPPORT_OBJECT[idx][1:]:
                        print(' '*len_to_object + filter_step)
        elif name.upper()=='HEADER':
            for idx in range(len(HEADER[0])):
                print(f'{idx+1})'.ljust(index_width) + f'{HEADER[0][idx]}')
        else:
            for idx in range(len(SUPPORT_OBJECT)):
                template(str(idx+1), HEADER[0][idx], SUPPORT_OBJECT[idx])


        del template, max_hdr_len, index_width, column_width, len_to_object













if __name__ == '__main__':

    # TEST MODULE

    this_module = gmn.get_module_name(str(sys.modules[__name__]))

    _columns = 5
    _rows = 3
    _orientation = 'COLUMN'

    DATA = np.random.uniform((_columns if _orientation=='COLUMN' else _rows, _rows if _orientation=='COLUMN' else _columns))

    DUM_HOLDER = np.empty((9,_columns), dtype=object)

    HEADER = np.fromiter((f'RUMPELSTILTSKIN COLUMN{_+1}' for _ in range(_columns)), dtype=object)
    VDTYPES = np.fromiter(('FLOAT' for _ in range(_columns)), dtype=object)
    MDTYPES = np.fromiter(('SPLIT_STR' for _ in range(_columns)), dtype=object)
    FILTERING = np.fromiter(([f'eggs', f'oatmeal', f'ramen'] for _ in range(_columns)), dtype=object)
    MIN_CUTOFFS = np.fromiter((None for _ in range(_columns)), dtype=object)
    USE_OTHER = np.fromiter(('N' for _ in range(_columns)), dtype=object)
    START_LAG = np.fromiter((None for _ in range(_columns)), dtype=object)
    END_LAG = np.fromiter((0 for _ in range(_columns)), dtype=object)
    SCALING = np.fromiter((np.char.split(f'these are the times that').tolist()), dtype=object)

    DUM_HOLDER[0] = HEADER
    DUM_HOLDER[1] = VDTYPES
    DUM_HOLDER[2] = MDTYPES
    DUM_HOLDER[3] = FILTERING
    DUM_HOLDER[4] = MIN_CUTOFFS
    DUM_HOLDER[5] = USE_OTHER
    DUM_HOLDER[6] = START_LAG
    DUM_HOLDER[7] = END_LAG
    DUM_HOLDER[8] = SCALING


    print(f'\n\n\n' + f'*' * 100)
    print(f'ApexPrintSupportObjects:')
    ApexPrintSupportObjects(
                            DATA,
                            'DATA',
                            DUM_HOLDER,
                            orientation=_orientation,
                            _columns=_columns,  # FOR SPEED, IF ALREADY AVAILABLE FROM CALLER
                            max_hdr_len=None,
                            calling_module=this_module,
                            calling_fxn='guard_test'
                            )


    print(f'\n\n\n' + f'*' * 100)
    print(f'PrintSupportObjects_NewObjects:')
    PrintSupportObjects_NewObjects(
                                    DATA,
                                    'DATA',
                                    DUM_HOLDER,
                                    orientation=_orientation,
                                    _columns=_columns,  # FOR SPEED, IF ALREADY AVAILABLE FROM CALLER
                                    max_hdr_len=None,
                                    calling_module=this_module,
                                    calling_fxn='guard_test'
                                    )

    print(f'\n\n\n' + f'*'*100)
    print(f'PrintSupportObjects_OldObjects:')
    PrintSupportObjects_OldObjects(
                                    DATA,
                                    'DATA',
                                    HEADER=HEADER,
                                    VALIDATED_DATATYPES=VDTYPES,
                                    MODIFIED_DATATYPES=MDTYPES,
                                    FILTERING=FILTERING,
                                    MIN_CUTOFFS=MIN_CUTOFFS,
                                    USE_OTHER=USE_OTHER,
                                    START_LAG=START_LAG,
                                    END_LAG=END_LAG,
                                    SCALING=SCALING,
                                    orientation=_orientation,
                                    _columns=_columns,  # FOR SPEED, IF ALREADY AVAILABLE FROM CALLER
                                    max_hdr_len=None,
                                    calling_module=this_module,
                                    calling_fxn='guard_test'
                                    )





    print(f'\n\n\n' + f'*'*100)
    print(f'PrintSingleSupportObjects:')
    NAMES = ['HEADER', 'VALIDATEDDATATYPES', 'MODIFIEDDATATYPES', 'FILTERING', 'MINCUTOFFS', 'USEOTHER', 'STARTLAG',
             'ENDLAG', 'SCALING']

    for OBJECT, name in zip(DUM_HOLDER, NAMES):
        print(f'\n\n\n{name} ********************************************* ')
        PrintSingleSupportObject(OBJECT, name, HEADER=HEADER, calling_module=this_module, calling_fxn='guard_test')







