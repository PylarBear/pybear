# Authors:
#
#       Bill Sousa
#
# License: BSD 3 clause


import pandas as pd, numpy as np, inspect
from copy import deepcopy
from data_validation import validate_user_input as vui
# from ML_PACKAGE._data_validation import ValidateObjectType as vot, validate_modified_object_type as vmot
import sparse_dict as sd


class IdentifyObjectAndPrint:

    """Pass any object and get a description and a view of it.

    Parameters
    ---------


    Attributes
    ---------


    See Also
    ------
    None


    Notes
    ----
    None


    Examples
    -------
    >>> from pybear.debug import _IdentifyObjectAndPrint





    """










    def __init__(
                    self,
                    OBJECT: any,
                    object_name: str,
                    module: str,
                    rows:int=1,
                    columns:int=1,
                    start_row:int=0,
                    start_col:int=0

    ):










        self.OBJECT = OBJECT
        self.object_name = object_name

        # GET MODULE / FXN STUFF ##############################################################
        self.module = module    # gmn.get_module_name(str(sys.modules[module]))
        # IF "module" NOT IN inspect STR, THEN VAR IN QSTN IS INSIDE A FXN
        if 'module' not in inspect.stack()[1][3]: self.fxn = inspect.stack()[1][3]
        else:  # IF "module" IS INSIDE inspect STR, THEN VAR IN QUESTION IS IN THE MODULE NOT IN A FXN
            self.fxn = 'None'
        ########################################################################################

        self.rows = rows
        self.columns = columns
        self.start_row = start_row
        self.start_col = start_col

        xx = self.OBJECT
        self.outer_type = self.ipt(xx)

        self.type_desc = vot.ValidateObjectType(xx).validate_object_type()[0]
        self.is_sparse_dict = self.is_it_a_sparse_dict()

        # GET LENGTH OF OBJECT ##########################################################################################
        while True:
            # 12-3-21 - DF LOGIC IS TOO ILLOGICAL, len1 & len2 ARE RECALCULATED FOR DF LATER AFTER vot FINDS
            # THAT IT IS A DF
            try: # FOR A LIST-TYPE OF LIST-TYPES
                if True in map(lambda x: x in self.ipt(xx), ('LIST', 'ARRAY', 'TUPLE')) and \
                    True in map(lambda y: y in self.ipt(xx[0]), ('LIST', 'ARRAY', 'TUPLE', 'SET')):
                    self.len1 = len(xx)
                    self.print_len1 = f'[:{len(xx)}]'  # GET OVERALL LEN WHILE HERE
                    break
            except:
                try: # FOR A SINGLE LIST-TYPE
                    if True in map(lambda x: x in self.ipt(xx), ('LIST', 'ARRAY', 'TUPLE', 'SET')):
                        self.len1 = float('inf') #len(xx)
                        self.print_len1 = f'' #[:{len(xx)}]'
                        break
                except: pass
                # print(f'Error trying to get length of {self.object_name}, trying to convert to string...')

            try:
                if isinstance(xx, dict):
                    if self.is_sparse_dict:
                        self.type_desc = f'SPARSE DICTIONARY OF {sd.outer_len(xx)} INNER DICTIONARIES OF LENGTH {sd.inner_len(xx)}'
                        self.len1 = sd.outer_len(xx)
                        self.print_len1 = f'[:{self.len1}]'
                    else:  # IF NOT A SPARSE DICT
                        self.len1 = len(list(xx.keys()))
                        self.print_len1 = f'[:{self.len1}]'

                    break
            except: pass

            try:
                if isinstance(xx, pd.DataFrame):
                    df_col_len = len(xx)
                    self.len1 = min(df_col_len, self.rows)
                    self.print_len1 = f'[:{df_col_len}]'
                    break
            except:
                pass

            try:
                if isinstance(xx, (str, int, float, bool)):
                    self.len1 = len(str(xx))
                    self.print_len1 = f'[:{len(str(xx))}]'
                    break
            except: pass

            print(f'Cannot get length of {self.object_name}')

            break
        # END GET LENGTH OF OBJECT ##########################################################################################

        # GET LENGTH OF OBJECT[0] #######################################################################################
        while True:
            try:  # WORKS FOR A LIST-TYPE OF LIST-TYPES
                if self.ipt(xx) in ['LIST', 'ARRAY', 'TUPLE'] and \
                            self.ipt(xx[0]) in ['LIST', 'ARRAY', 'TUPLE', 'SET']:
                    self.len2 = len(xx[0])
                    self.print_len2 = f'[:{len(xx[0])}]'   # GET OVERALL LEN OF [0] WHILE HERE
                    break
            except:
                try:  # WORKS FOR A SINGLE LIST-TYPE
                    if self.ipt(xx) in ['LIST', 'ARRAY', 'TUPLE', 'SET']:
                        self.len2 = len(xx)
                        self.print_len2 = f'[:{self.len2}]'
                        break
                except: pass

            # WORKS FOR DICT
            try:
                if isinstance(xx, dict):
                    if self.is_sparse_dict:
                        self.len2 = sd.inner_len(xx)
                        self.print_len2 = f'[:{self.len2}]'
                    else:  # IF NOT A SPARSE DICT
                        self.len2 = float('inf')
                        self.print_len2 = f'[N/A]'
                    break
            except: pass

            # DONT NEED FOR DATAFRAME, HANDLED ELSEWHERE
            if isinstance(xx, pd.DataFrame):
                self.lens_of_df(xx)

            try:  # WORKS FOR STR, FLOAT, BIN, INT, BOOL
                self.len2 = float('inf')
                self.print_len2 = '[N/A]'
            except:
                print(f'Cannot get length of {self.object_name}[0]')

            break
        # END GET LENGTH OF OBJECT[0] #######################################################################################

        # CALCULATE start_row, end_row, start_col, end_col
        self.get_bounds()   # IF NOT A DF, GET BOUNDS AS PART OF __init__


    def ipt(self, _):     #
        '''Identify Python type'''
        return vmot.validate_modified_object_type(_)


    def is_it_a_sparse_dict(self):
        return sd.is_sparse_dict(self.OBJECT)


    def lens_of_df(self, DF):
        self.len1 = len([_ for _ in DF])
        self.print_len1 = f'[:{len([_ for _ in DF])}]'  # GET OVERALL LEN WHILE HERE
        self.len2 = len(DF)
        self.print_len2 = f'[:{len(DF)}]'  # GET OVERALL LEN WHILE HERE
        self.get_bounds()


    def get_bounds(self):
        # 12-29-21 IF COL OR ROW WINDOW CREATED BY USER (THRU start_col + columns, start_row + rows)
        # PUTS THE PRINT WINDOW OUT OF THE DATA RANGE, JUST PRINT #rows / #columns AT EDGE OF DATA
        # THIS RUNS AS IN __init__, BUT IF OBJ IS LATER FOUND TO BE A DF, THIS FXN IS CALLED AGAIN
        self.columns = min(self.columns, self.len1)  # NO. COLUMNS IN WINDOW MUST BE <= TOTAL COLUMNS
        self.rows = min(self.rows, self.len2)  # NO. ROWS IN WINDOW MUST <= TOTAL ROWS
        self.end_col = min(self.len1, self.start_col + self.columns)
        self.end_row = min(self.len2, self.start_row + self.rows)
        self.start_col = min(self.start_col, max(0, self.len1 - self.columns))
        self.start_row = min(self.start_row, max(0, self.len2 - self.rows))


    def run(self):

        xx = self.OBJECT

        print(f'MODULE = {self.module}   FXN = {self.fxn}   OBJECT = {self.object_name}')
        print(f'"{self.object_name}" IS A(N) {self.type_desc}')

        # TO MAKE HANDLING "LIST-TYPE OF LIST-TYPE" EASIER
        TYPE_FINDER = lambda type_str, OPTIONS: [_ in type_str for _ in OPTIONS]

        if 'FRAME' in self.outer_type:  # WHEN IT'S A DATAFRAME
            self.lens_of_df(xx)

            print(f'{self.object_name}[{self.start_col}:{self.end_col}][{self.start_row}:{self.end_row}] OF {self.print_len1}{self.print_len2} LOOKS LIKE: ')
            print(xx[[_ for _ in xx][self.start_col:self.end_col]].iloc[self.start_row:self.end_row])

        # LIST-TYPE OF LIST-TYPES
        # HAVE TO DO IT LIKE THIS BECAUSE self.type_desc returns "LIST OF X LIST(S)" SO INSTEAD OF TRYING TO
        # ACCOMMODATE "X" IN THAT TEXT, JUST LOOK AT THE PYTHON TYPES AGAIN
        elif True in TYPE_FINDER(str(type(xx)).upper(),  ['LIST', 'ARRAY', 'TUPLE']) and len(xx) > 0 and \
                True in TYPE_FINDER(str(type(xx[0])).upper(), ['LIST', 'ARRAY', 'TUPLE', 'SET']):

            print(f'{self.object_name}[{self.start_col}:{self.end_col}][{self.start_row}:{self.end_row}] of {self.print_len1}{self.print_len2} LOOKS LIKE: ')
            print(f"{['[','[','('][np.argwhere(np.array(TYPE_FINDER(str(type(xx)).upper(), ['LIST', 'ARRAY', 'TUPLE']))==True)[0][0]]}")
            [print(xx[_][self.start_row:self.end_row]) for _ in range(self.start_col, self.end_col)]
            print(f"{[']',']',')'][np.argwhere(np.array(TYPE_FINDER(str(type(xx)).upper(), ['LIST', 'ARRAY', 'TUPLE']))==True)[0][0]]}")

        # SINGLE LIST-TYPE
        elif True in TYPE_FINDER(self.outer_type, ['LIST', 'ARRAY', 'TUPLE']):
            print(f'{self.object_name}[{self.start_row}:{self.end_row}] of {self.print_len1} LOOKS LIKE: ')
            print(xx[self.start_row:self.end_row])

        # LIST-TYPE OF SETS
        elif True in TYPE_FINDER(str(type(xx)).upper(), ['LIST', 'ARRAY', 'TUPLE']) and len(xx) > 0 and \
                'SET' in str(type(xx[0])).upper():
            print(f'{self.object_name}[{self.start_col}:{self.end_col}][{self.start_row}:{self.end_row}] of {self.print_len1}{self.print_len2} LOOKS LIKE: ')
            print(f"{['[','[','('][np.argwhere(np.array(TYPE_FINDER(str(type(xx)).upper(), ['LIST', 'ARRAY', 'TUPLE']))==True)[0][0]]}")
            [print(xx[_][self.start_row:self.end_row]) for _ in range(self.start_col, self.end_col)]
            print(f"{[']',']',')'][np.argwhere(np.array(TYPE_FINDER(str(type(xx)).upper(), ['LIST', 'ARRAY', 'TUPLE']))==True)[0][0]]}")

        # SINGLE SET (CANT HAVE A SET OF SETS)
        elif 'SET' in self.outer_type:
            if vui.validate_user_str(f'\nA SET IS NOT SUBSCRIPTABLE, MUST PRINT THE ENTIRE OBJECT.  PROCEED? (y/n) > ', 'YN') == 'Y':
                print(f'{self.object_name}{self.print_len1} LOOKS LIKE: ')
                print(xx)

        elif 'DICT' in self.outer_type:
            print(f'{self.object_name}[{self.start_col}:{self.end_col}] of {self.print_len1} LOOKS LIKE: ')
            ENTRIES = [i for i in xx][self.start_col:self.end_col]
            [print(f'{_}: {str(xx[_])[:100]}') for _ in ENTRIES]

        # CHAR SEQ
        elif True in map(lambda x: x in self.outer_type, ('STR', 'FLOAT', 'INT', 'BOOL', 'BIN')):
            print(f'{self.object_name}[{self.start_col}:{self.end_col}] of {self.print_len1} LOOKS LIKE: ')
            print(f'{str(xx)[self.start_col:self.end_col]}')

        else:
            print(f'{self.type_desc}')


    def DF_print(self, DF, orientation):

        '''in __init__, len1 is always number of inner objects, len2 is always len of inner objects.
            when user specifies orientation in run_print_as_df, it is specifying the orientation of the inner objects, i.e.,
            "column" means the inner objects hold columns of data, and "row" means holding rows of data.
            but when specifying which rows / columns to print, it is not based on the orientation, that is, if saying print 10
            rows that are oriented as rows, it prints within the inner objects.  If saying print 10 rows when oriented as
            columns,  it cuts across the inner objects.
            So to accomplish this, since __init__ always reads len1 and len2 as stated above, and start / end row / col
            is based on these measurements and row / col can change, charting and caption code is written such that len2
            always represents len of inner objects.  So if in "column" orientation, flip all measurements.
        '''

        self.lens_of_df(DF)  # DO THIS FIRST BEFORE <FLIPPING len / row / col WHEN IN "COLUMN" ORIENTATION!!!!>
        self.get_bounds()
        # if orientation == 'column':
        #     self.len1, self.len2, self.print_len1, self.print_len2 = self.len2, self.len1, self.print_len2, self.print_len1
        #     self.start_col, self.end_col, self.start_row, self.end_row = self.start_row, self.end_row, self.start_col, self.end_col

        print(f'\nMODULE = {self.module}   FXN = {self.fxn}   OBJECT = {self.object_name}')
        print(f'"{self.object_name}" IS A(N) {self.type_desc}')

        display_txt = f'{self.object_name}[{self.start_row}:{self.end_row}][{self.start_col}:{self.end_col}]'
        print(f'AS A DF, {display_txt} OF ' + f'{self.print_len2}{self.print_len1} LOOKS LIKE: ')
        print(DF[[_ for _ in DF][self.start_col:self.end_col]].iloc[self.start_row:self.end_row])


    def run_print_as_df(self, df_columns='', orientation='column'):
        # TRANSPOSE BELOW TO CONVERT TO [] = ROWS FOR DF)
        while True:

            if orientation.lower() not in ['column', 'row']:
                raise ValueError(f'\n****INVALID orientation IN IdentifyObjectAndPrint****\n')


            elif True in map(lambda x: x in self.outer_type, ('STR', 'FLOAT', 'INT', 'BOOL', 'BIN')):
                self.run()
                break


            if 'FRAME' in str(type(self.OBJECT)).upper():
                if orientation == 'row':
                    df_data = [np.array(self.OBJECT[_]) for _ in self.OBJECT]
                    INDEX = list(self.OBJECT.keys())
                    df_columns = df_columns
                    NEW_DF = pd.DataFrame(data=df_data, index=INDEX, columns=df_columns)
                    self.type_desc = vot.ValidateObjectType(NEW_DF).validate_object_type()[0]
                    del NEW_DF

                elif orientation == 'column':
                    df_data = np.array([np.array(self.OBJECT[_]) for _ in self.OBJECT]).transpose()
                    INDEX = np.array([_ for _ in range(len(df_data))], dtype=int)
                    df_columns = df_columns


            elif 'DICT' in self.outer_type:
                if self.is_sparse_dict:
                    if orientation == 'row':
                        df_data = deepcopy(self.OBJECT)
                    elif orientation == 'column':
                        df_data = sd.sparse_transpose(deepcopy(self.OBJECT))

                    df_data = sd.unzip_to_ndarray(df_data)[0]
                    # df_columns = df_columns
                    INDEX = list(range(len(df_data)))
                else:  # IF NOT A SPARSE DICT
                    if orientation == 'row':
                        df_data = np.array([np.array(self.OBJECT[_]) for _ in self.OBJECT])
                        INDEX = list(self.OBJECT.keys())
                        df_columns = [self.object_name]
                    elif orientation == 'column':
                        df_data = np.array([np.array(self.OBJECT[_]) for _ in self.OBJECT]).transpose()
                        INDEX = [*range(len(df_data))]
                        df_columns = df_columns


            elif 'SET' in self.outer_type:
                if orientation == 'column':
                    df_data = [[_] for _ in self.OBJECT]
                    INDEX = np.arange(0, len(self.OBJECT))
                    df_columns = df_columns
                elif orientation == 'row':
                    df_data = np.array([self.OBJECT])
                    INDEX = [f'{self.object_name}']
                    df_columns = df_columns


            elif True in map(lambda x: x in self.outer_type, ('LIST', 'TUPLE', 'ARRAY')):
                if orientation == 'column':
                    df_data = np.array(self.OBJECT, dtype=object).transpose()  # TRANSPOSE TO [] = ROWS
                    INDEX = np.arange(0, len(df_data))

                elif orientation == 'row':
                    df_data = np.array(self.OBJECT, dtype=object)
                    INDEX = np.arange(0, len(df_data))

            try:
                DF = pd.DataFrame(data=df_data, columns=df_columns, index=INDEX, dtype=object).fillna(0).sort_index()
                self.DF_print(DF, orientation)
            except:
                print(f'\n*** ERROR TRYING TO BUILD DF FOR {self.object_name} WITH INDEX.  ATTEMPTING WITHOUT INDEX.***')
                try:
                    DF = pd.DataFrame(data=df_data, columns=df_columns, dtype=object)
                    self.DF_print(DF, orientation)
                except:
                    print(f'\n*** ERROR TRYING TO BUILD DF FOR {self.object_name} WITHOUT INDEX.  UNABLE TO PRINT OBJECT AS DF.***\n')
                    print(f'COULD NOT PRINT AS DF, PRINTING AS IS:')
                    self.run()

            break



if __name__ == '__main__':
    dum_rows = 300
    dum_cols = 25
    data = np.random.rand(0,10,(dum_cols,dum_rows))  # LIST OF LISTS (CAN BE TURNED INTO NUMPY OF NUMPYS)
    dum_str = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'   # CAN BE USED AS STR (BEING USED AS HEADER FOR DF, DONT DELETE)
    header = [dum_str[x] for x in range(dum_cols)]
    TEST_DF = pd.DataFrame(columns=header, data=np.array(data).transpose())  # DATAFRAME
    import uuid
    DICT = dict() #DICTIONARY
    LIST_OF_STR = []    # LIST OF STR
    for char in dum_str:
        _ = str(uuid.uuid4())
        DICT[char] = _  # BUILD DICT
        LIST_OF_STR.append(_)  # BUILD LIST OF STR

    TEST_HEADER = np.array(['COL ' + str(_) for _ in range(dum_cols)], dtype=object)

    # TEST_DATA = np.array([[f'{str(_)}-{str(__)}' for __ in range(dum_rows)] for _ in range(dum_cols)], dtype=object)
    TEST_DATA = sd.create_random((dum_cols, dum_rows), 60)

    IdentifyObjectAndPrint([], 'DATA', __name__, rows=10, columns=5, start_row=0, start_col=0).run_print_as_df(df_columns=[], orientation='column')











