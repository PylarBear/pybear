

def OBJ_TYPES():
    return  {'STR': 'STR',
             'INT': 'INT',
             'BIN': 'BIN',
             'FLOAT': 'FLOAT',
             'BOOL': 'BOOL',
             '.DATETIME': 'DATETIME', # 10/27/21 LOOKS LIKE ALL NUMPY DATE/TIME/DATETIME FXNS ARE HANDLED BY n.datetime64
             '.DATE': 'DATE',                           # SO IT LOOKS LIKE py AND n DATETIME CAN BE CAPTURED IN THIS ONE LOOKUP
             '.TIME': 'TIME',  # PANDAS DATES pandas._libs.tslibs.timestamps.Timestamp
             'TIMESTAMP': 'DATETIME',
             'NAT': 'DATETIME', # 12-1-21 ACCOMMODATE THAT WHEN A n.array dtype IS datetime & IT HAS A "BAD" ENTRY, n RETURNS "NaT"
             'ARRAY': 'ARRAY',
             'DATAFRAME': 'DATAFRAME',
             'SERIES': 'DATAFRAME COLUMN',
             # A PANDAS DATAFRAME COLUMN IS CALLED A SERIES, BUT WANT TO CALL IT 'DATAFRAME COLUMN'
             'DICT': 'DICTIONARY',  #REMEMBER THAT THIS USED TO BE 'DICT' BUT CHANGED IT TO 'DICTIONARY' ON 11-27-2021, WITHOUT EXHAUSTIVE ASSESSMENT OF THE REPERCUSSIONS
             'LIST': 'LIST',
             'SET': 'SET',
             'TUPLE': 'TUPLE',
             'FUNCTION': 'FUNCTION'
             }


# CALLED BY ValidateObjectType, ValidateList
def list_of_charseqs():
    return ['STR', 'INT', 'BIN', 'FLOAT', 'CURRENCY', 'DATE', 'TIME', 'DATETIME', 'BOOL']


# CALLED BY ValidateObjectType, ValidateList
def list_of_listtypes():
    return ['ARRAY', 'SET', 'TUPLE', 'LIST']


# CALLED BY ValidateObjectType, ValidateList, IdentifyObjectAndPrint, ETC.
def validate_modified_object_type(_):  # LOOPING IT OUT LIKE THIS TO UTILIZE DICTIONARY OF TYPES

    __ = str(type(_)).upper()

    for item in OBJ_TYPES():
        if item.upper() in __:
            if item.upper() == 'INT' and (_ == 0 or _ == 1):
                item = 'BIN'
            modified_object_type = OBJ_TYPES()[item].upper()
            break
    else:
        modified_object_type = 'UNRECOGNIZED OBJECT TYPE'
        print(f'DATA LOOKS LIKE = {_}')

    return modified_object_type




if __name__ == '__main__':
    pass



