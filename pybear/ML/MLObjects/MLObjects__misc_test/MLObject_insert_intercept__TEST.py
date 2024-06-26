import sys, inspect, time
from MLObjects import MLObject as mlo
import numpy as np
import sparse_dict as sd
from copy import deepcopy
from debug import get_module_name as gmn
from MLObjects.TestObjectCreators import test_header as th
from MLObjects.SupportObjects import BuildFullSupportObject as bfso
from general_sound import winlinsound as wls


print_green = lambda words: print(f'\033[92m{words}\033[0m')
print_red = lambda words: print(f'\033[91m{words}\033[0m')

calling_module = gmn.get_module_name(str(sys.modules[__name__]))


##########################################################################################################################
# TEST DATA VALIDATION EXCEPTIONS ########################################################################################

test_name = 'data validation exceptions'
print(f'Running {test_name}...')


_rows, _cols = 100, 80
given_orientation = 'COLUMN'

DATA = np.random.randint(1,10, (_cols if given_orientation=='COLUMN' else _rows, _rows if given_orientation=='COLUMN' else _cols))


TestClass1 = mlo.MLObject(
                          DATA,
                          given_orientation,
                          name="DATA",
                          return_orientation='AS_GIVEN',
                          return_format='AS_GIVEN',
                          bypass_validation=False,
                          calling_module=calling_module,
                          calling_fxn=test_name
)

BASE_KWARGS = {
               'col_idx': _cols,
               'COLUMN_OR_VALUE_TO_INSERT': 1,
               'insert_orientation': None,
               'HEADER_OR_FULL_SUPOBJ': None,
               'SUPOBJ_INSERT_VECTOR':None,
               'datatype_string': None,
               'header_string': None,
               'CONTEXT': None
}

fails = 0

# TestClass1.insert_intercept(col_idx, COLUMN_OR_VALUE_TO_INSERT, insert_orientation=None, HEADER_OR_FULL_SUPOBJ=None,
#                      SUPOBJ_INSERT_VECTOR=None, datatype_string=None, header_string=None, CONTEXT=None)


# col_idx       # HANDLED EXCLUSIVELY BY insert_column() #############################################################
print_green(f'\nTesting negative col_idx excepts...')
KWARGS = deepcopy(BASE_KWARGS); KWARGS['col_idx'] = -1
try: TestClass1.insert_intercept(**KWARGS); fails+=1; print_red(f'Fail. Did not except for negative col_idx.')
except: print_green(f'Pass.')


print_green(f'\nTesting col_idx > cols excepts...')
KWARGS = deepcopy(BASE_KWARGS); KWARGS['col_idx'] = _cols+1
try: TestClass1.insert_intercept(**KWARGS); fails+=1; print_red(f'Fail. Did not except for col_idx > cols.')
except: print_green(f'Pass.')
# END col_idx       # HANDLED EXCLUSIVELY BY insert_column() #########################################################


# COLUMN_OR_VALUE_TO_INSERT & insert_orientation ################################################################
print_green(f'\nTesting COLUMN_OR_VALUE_TO_INSERT other than number or list excepts...')
KWARGS = deepcopy(BASE_KWARGS); KWARGS['COLUMN_OR_VALUE_TO_INSERT'] = 'DUM'
try: TestClass1.insert_intercept(**KWARGS); fails+=1; print_red(f'Fail. COLUMN_OR_VALUE_TO_INSERT other than number or list did not except.')
except: print_green(f'Pass.')


print_green(f'\nTesting COLUMN_OR_VALUE_TO_INSERT passed as number does not require insert_orientation...')
KWARGS = deepcopy(BASE_KWARGS)
KWARGS['COLUMN_OR_VALUE_TO_INSERT'] = 1
KWARGS['insert_orientation'] = None
try:
    TestClass1.insert_intercept(**KWARGS)
    print_green(f'Pass.')
    KWARGS['HEADER_OR_FULL_SUPOBJ'] = TestClass1.HEADER_OR_FULL_SUPOBJ
    KWARGS['CONTEXT'] = TestClass1.CONTEXT
except: fails+=1; print_red(f'Fail. COLUMN_OR_VALUE_TO_INSERT passed as number excepted with insert_orientation==None.')

# DELETE SUCCESSFULLY INSERTED INTERCEPT TO GO BACK TO ORIGINAL STATE
TestClass1.delete_columns([KWARGS['col_idx']], KWARGS['HEADER_OR_FULL_SUPOBJ'], KWARGS['CONTEXT'])


print_green(f'\nTesting COLUMN_OR_VALUE_TO_INSERT as vector requires insert_orientation...')
KWARGS = deepcopy(BASE_KWARGS)
KWARGS['COLUMN_OR_VALUE_TO_INSERT'] = np.ones(_rows)
KWARGS['insert_orientation'] = None
try: TestClass1.insert_intercept(**KWARGS); fails+=1; print_red(f'Fail. COLUMN_OR_VALUE_TO_INSERT as vector allowed insert_orientation as None.')
except: print_green(f'Pass.')


print_green(f'\nTesting COLUMN_OR_VALUE_TO_INSERT as vector excepts for insert_orientation other than "ROW" OR "COLUMN"...')
KWARGS = deepcopy(BASE_KWARGS)
KWARGS['COLUMN_OR_VALUE_TO_INSERT'] = np.ones(_rows)
KWARGS['insert_orientation'] = 'DUM'
try: TestClass1.insert_intercept(**KWARGS); fails+=1; print_red(f'Fail. COLUMN_OR_VALUE_TO_INSERT as vector allowed insert_orientation other than "ROW" OR "COLUMN".')
except: print_green(f'Pass.')


print_green(f'\nTesting COLUMN_OR_VALUE_TO_INSERT as vector insert_orientation accepts "ROW"...')
KWARGS = deepcopy(BASE_KWARGS)
KWARGS['COLUMN_OR_VALUE_TO_INSERT'] = np.ones((_rows,1))
KWARGS['insert_orientation'] = 'ROW'
try:
    TestClass1.insert_intercept(**KWARGS)
    print_green(f'Pass.')
    KWARGS['HEADER_OR_FULL_SUPOBJ'] = TestClass1.HEADER_OR_FULL_SUPOBJ
    KWARGS['CONTEXT'] = TestClass1.CONTEXT
except: fails+=1; print_red(f'Fail. COLUMN_OR_VALUE_TO_INSERT as vector excepted when insert_orientation=="ROW".')

# DELETE SUCCESSFULLY INSERTED INTERCEPT TO GO BACK TO ORIGINAL STATE
TestClass1.delete_columns([KWARGS['col_idx']], KWARGS['HEADER_OR_FULL_SUPOBJ'], KWARGS['CONTEXT'])


print_green(f'\nTesting COLUMN_OR_VALUE_TO_INSERT as vector accepts "COLUMN"...')
KWARGS = deepcopy(BASE_KWARGS)
KWARGS['COLUMN_OR_VALUE_TO_INSERT'] = np.ones(_rows)
KWARGS['insert_orientation'] = 'COLUMN'
try: TestClass1.insert_intercept(**KWARGS); print_green(f'Pass.')
except: fails+=1; print_red(f'Fail. COLUMN_OR_VALUE_TO_INSERT as vector excepted when insert_orientation=="COLUMN".')

# DELETE SUCCESSFULLY INSERTED INTERCEPT TO GO BACK TO ORIGINAL STATE
TestClass1.delete_columns([KWARGS['col_idx']], KWARGS['HEADER_OR_FULL_SUPOBJ'], KWARGS['CONTEXT'])


print_green(f'\nTesting COLUMN_OR_VALUE_TO_INSERT as vector too short excepts...')
KWARGS = deepcopy(BASE_KWARGS)
KWARGS['COLUMN_OR_VALUE_TO_INSERT'] = np.ones(_rows-1)
KWARGS['insert_orientation'] = 'COLUMN'
try: TestClass1.insert_intercept(**KWARGS); fails+=1; print_red(f'Fail. COLUMN_OR_VALUE_TO_INSERT as vector did not except for short vector.')
except: print_green(f'Pass.')


print_green(f'\nTesting COLUMN_OR_VALUE_TO_INSERT as vector too long excepts...')
KWARGS = deepcopy(BASE_KWARGS)
KWARGS['COLUMN_OR_VALUE_TO_INSERT'] = np.ones(_rows+1)
KWARGS['insert_orientation'] = 'COLUMN'
try: TestClass1.insert_intercept(**KWARGS); fails+=1; print_red(f'Fail. COLUMN_OR_VALUE_TO_INSERT as vector did not except for long vector.')
except: print_green(f'Pass.')


print_green(f'\nTesting COLUMN_OR_VALUE_TO_INSERT excepts if filled with different numbers (not a constant)...')
KWARGS = deepcopy(BASE_KWARGS)
KWARGS['COLUMN_OR_VALUE_TO_INSERT'] = np.ones(_rows); KWARGS['COLUMN_OR_VALUE_TO_INSERT'][0] = 5
KWARGS['insert_orientation'] = 'COLUMN'
try: TestClass1.insert_intercept(**KWARGS); fails+=1; print_red(f'Fail. COLUMN_OR_VALUE_TO_INSERT as vector did not except for non-constant numbers.')
except: print_green(f'Pass.')
# END COLUMN_OR_VALUE_TO_INSERT & insert_orientation ############################################################





# HEADER_OR_FULL_SUPOBJ, SUPOBJ_INSERT_VECTOR, datatype_string, & header_string #################################

# SET BASE_KWARGS COLUMN_OR_VALUE_TO_INSERT AND insert_orientation PERMANENTLY TO TEST SUPOBJ THINGS
BASE_KWARGS['COLUMN_OR_VALUE_TO_INSERT'] = np.ones(_rows)
BASE_KWARGS['insert_orientation'] = 'COLUMN'

# TESTING FOR NO HEADER_OR_FULL_SUPOBJ PASSED
for trial in ("SUPOBJ_INSERT_VECTOR", "datatype_string", "header_string"):
    print_green( f'\nTesting excepts when HEADER_OR_FULL_SUPOBJ is not passed and only {trial} is passed...')
    KWARGS = deepcopy(BASE_KWARGS)
    if trial == "SUPOBJ_INSERT_VECTOR": KWARGS["SUPOBJ_INSERT_VECTOR"] = 'DUM'
    elif trial == "datatype_string": KWARGS["datatype_string"] = 'DUM'
    elif trial == "header_string": KWARGS["header_string"] = 'DUM'
    try: TestClass1.insert_intercept(**KWARGS); fails+=1; print_red(f'Fail. Did not except for {trial} passed when HEADER_OR_FULL_SUPOBJ not passed.')
    except: print_green(f'Pass.')


# TESTING FOR HEADER_OR_FULL_SUPOBJ PASSED
print_green(f'\nTesting HEADER_OR_FULL_SUPOBJ excepts if not sized as header or a full support object...')
KWARGS = deepcopy(BASE_KWARGS)
KWARGS['HEADER_OR_FULL_SUPOBJ'] = np.tile(th.test_header(_cols), (3,1))
try:
    TestClass1.insert_intercept(**KWARGS)
    fails+=1
    print_red(f'Fail. HEADER_OR_FULL_SUPOBJ did not except if not sized correctly for a header or a full support object.')
except: print_green(f'Pass.')


# TESTS FOR HEADER_OR_FULL_SUPOBJ AS HEADER
for trial in ("single", "double"):
    single_double = "[]" if trial=="single" else "[[]]"
    print_green(f'\nTesting HEADER_OR_FULL_SUPOBJ as header allows {single_double} format...')
    KWARGS = deepcopy(BASE_KWARGS)
    KWARGS['HEADER_OR_FULL_SUPOBJ'] = th.test_header(_cols)[0] if trial=="single" else th.test_header(_cols)
    KWARGS['header_string'] = 'TEST INTERCEPT'
    try:
        TestClass1.insert_intercept(**KWARGS)
        KWARGS['HEADER_OR_FULL_SUPOBJ'] = TestClass1.HEADER_OR_FULL_SUPOBJ
        KWARGS['CONTEXT'] = TestClass1.CONTEXT
        print_green(f'Pass.')
        # DELETE SUCCESSFULLY INSERTED INTERCEPT TO GO BACK TO ORIGINAL STATE
        try: TestClass1.delete_columns([KWARGS['col_idx']], KWARGS['HEADER_OR_FULL_SUPOBJ'], KWARGS['CONTEXT'])
        except:
            print(f"[KWARGS['col_idx']] = {[KWARGS['col_idx']]}")
            print(f"KWARGS['HEADER_OR_FULL_SUPOBJ'] = {KWARGS['HEADER_OR_FULL_SUPOBJ']}")
            print(f"KWARGS['CONTEXT'] = {KWARGS['CONTEXT']}")
            raise ValueError
    except ValueError: raise Exception(f'delete_columns excepted when trying to delete inserted intercept.')
    except: fails+=1; print_red(f'Fail. HEADER_OR_FULL_SUPOBJ excepted when passed as a header in {single_double} format.')
del single_double


print_green(f'\nTesting HEADER_OR_FULL_SUPOBJ as header only excepts if not enough columns...')
KWARGS = deepcopy(BASE_KWARGS)
KWARGS['HEADER_OR_FULL_SUPOBJ'] = th.test_header(_cols)[..., :-1]
try: TestClass1.insert_intercept(**KWARGS); fails+=1; print_red(f'Fail. HEADER_OR_FULL_SUPOBJ did not except when not enough columns.')
except: print_green(f'Pass.')


print_green(f'\nTesting HEADER_OR_FULL_SUPOBJ as header only excepts if too many columns...')
KWARGS = deepcopy(BASE_KWARGS)
KWARGS['HEADER_OR_FULL_SUPOBJ'] = np.insert(th.test_header(_cols), 0, 'EXTRA COLUMN', axis=1)
try: TestClass1.insert_intercept(**KWARGS); fails+=1; print_red(f'Fail. HEADER_OR_FULL_SUPOBJ did not except when too many columns.')
except: print_green(f'Pass.')


# SET BASE_KWARGS HEADER_OR_FULL_SUPOBJ TEMPORARILY AS A HEADER ONLY TO TEST SUPOBJ THINGS
BASE_KWARGS["HEADER_OR_FULL_SUPOBJ"] = th.test_header(_cols)


print_green(f'\nTesting HEADER_OR_FULL_SUPOBJ as header excepts if full SUPOBJ_INSERT_VECTOR PASSED...')
KWARGS = deepcopy(BASE_KWARGS)
KWARGS["SUPOBJ_INSERT_VECTOR"] = ['TEST INTERCEPT','INT','INT',[],0,'N',0,0,'']
try:
    TestClass1.insert_intercept(**KWARGS)
    fails+=1
    print_red(f'Fail. HEADER_OR_FULL_SUPOBJ did not except when passed a full SUPOBJ_INSERT_VECTOR to a HEADER_OR_FULL_SUPOBJ as header.')
except: print_green(f'Pass.')


for trial in ["single", "double"]:
    single_double = "[]" if trial=="single" else "[[]]"
    print_green(f'\nTesting HEADER_OR_FULL_SUPOBJ as header does not except if SUPOBJ_INSERT_VECTOR as {single_double} is PASSED...')
    KWARGS = deepcopy(BASE_KWARGS)
    KWARGS["SUPOBJ_INSERT_VECTOR"] = ['TEST INTERCEPT'] if trial=="single" else [['TEST INTERCEPT']]
    try:
        TestClass1.insert_intercept(**KWARGS)
        KWARGS['HEADER_OR_FULL_SUPOBJ'] = TestClass1.HEADER_OR_FULL_SUPOBJ
        KWARGS['CONTEXT'] = TestClass1.CONTEXT
        print_green(f'Pass.')
        # DELETE SUCCESSFULLY INSERTED INTERCEPT TO GO BACK TO ORIGINAL STATE
        try:
            TestClass1.delete_columns([KWARGS['col_idx']], KWARGS['HEADER_OR_FULL_SUPOBJ'], KWARGS['CONTEXT'])
        except:
            print(f"[KWARGS['col_idx']] = {[KWARGS['col_idx']]}")
            print(f"KWARGS['HEADER_OR_FULL_SUPOBJ'] = {KWARGS['HEADER_OR_FULL_SUPOBJ']}")
            print(f"KWARGS['CONTEXT'] = {KWARGS['CONTEXT']}")
            raise ValueError
    except ValueError: raise Exception(f'delete_columns excepted when trying to delete inserted intercept.')
    except: fails+=1; print_red(f'Fail. HEADER_OR_FULL_SUPOBJ as header excepted when passed a SUPOBJ_INSERT_VECTOR as {single_double}.')
    del single_double


print_green(f'\nTesting excepts when simultaneously passed SUPOBJ_INSERT_VECTOR, header_string, datatype_string...')
KWARGS = deepcopy(BASE_KWARGS)
KWARGS["SUPOBJ_INSERT_VECTOR"] = ['TEST INTERCEPT']
KWARGS["header_string"] = 'TEST INTERCEPT'
KWARGS["datatype_string"] = 'INT'
try:
    TestClass1.insert_intercept(**KWARGS)
    fails+=1
    print_red(f'Fail. Did not except when simultaneously passed SUPOBJ_INSERT_VECTOR, header_string, datatype_string.')
except: print_green(f'Pass.')


print_green(f'\nTesting excepts when none of SUPOBJ_INSERT_VECTOR, header_string, or datatype_string are passed ...')
KWARGS = deepcopy(BASE_KWARGS)
try:
    TestClass1.insert_intercept(**KWARGS)
    fails+=1
    print_red(f'Fail. Did not except when none of SUPOBJ_INSERT_VECTOR, header_string, datatype_string were passed.')
except: print_green(f'Pass.')


print_green(f'\nTesting ignores datatype_string when passed header_string and datatype_string for header-only HEADER_OR_FULL_SUPOBJ...')
KWARGS = deepcopy(BASE_KWARGS)
KWARGS["header_string"] = 'TEST INTERCEPT'
KWARGS["datatype_string"] = 'INT'
try:
    TestClass1.insert_intercept(**KWARGS)
    KWARGS['HEADER_OR_FULL_SUPOBJ'] = TestClass1.HEADER_OR_FULL_SUPOBJ
    KWARGS['CONTEXT'] = TestClass1.CONTEXT
    print_green(f'Pass.')
except: fails+=1; print_red(f'Fail. Excepted and did not ignore datatype_string when header_string passed for HEADER_OR_FULL_SUPOBJ as header-only.')

# DELETE SUCCESSFULLY INSERTED INTERCEPT TO GO BACK TO ORIGINAL STATE
TestClass1.delete_columns([KWARGS['col_idx']], KWARGS['HEADER_OR_FULL_SUPOBJ'], KWARGS['CONTEXT'])


print_green(f'\nTesting excepts when not passed header_string for header-only HEADER_OR_FULL_SUPOBJ...')
KWARGS = deepcopy(BASE_KWARGS)
KWARGS["datatype_string"] = 'INT'
try: TestClass1.insert_intercept(**KWARGS); fails+=1; print_red(f'Fail. Did not except when not passed header_string for header-only HEADER_OR_FULL_SUPOBJ.')
except: print_green(f'Pass.')


print_green(f'\nTesting excepts when header_string passed as a non-string for header-only HEADER_OR_FULL_SUPOBJ...')
KWARGS = deepcopy(BASE_KWARGS)
KWARGS["datatype_string"] = ['INT']
try: TestClass1.insert_intercept(**KWARGS); fails+=1; print_red(f'Fail. Did not except when passed header_string as a non-string header-only HEADER_OR_FULL_SUPOBJ.')
except: print_green(f'Pass.')


# TESTS FOR HEADER_OR_FULL_SUPOBJ AS FULL_SUPOBJ
# CREATE A FULL SUPOBJ FOR TESTING
FULL_SUPOBJ = bfso.BuildFullSupportObject(OBJECT=DATA, object_given_orientation=given_orientation,
                          OBJECT_HEADER=th.test_header(_cols), SUPPORT_OBJECT=None, columns=_cols, quick_vdtypes=False,
                          MODIFIED_DATATYPES=None, print_notes=False, prompt_to_override=False, bypass_validation=True,
                          calling_module=calling_module, calling_fxn=test_name
).SUPPORT_OBJECT


BASE_KWARGS['HEADER_OR_FULL_SUPOBJ'] = FULL_SUPOBJ; del FULL_SUPOBJ


print_green(f'\nTesting HEADER_OR_FULL_SUPOBJ as full excepts if not enough columns...')
KWARGS = deepcopy(BASE_KWARGS)
KWARGS['HEADER_OR_FULL_SUPOBJ'] = np.delete(KWARGS['HEADER_OR_FULL_SUPOBJ'], 0, axis=1)
try: TestClass1.insert_intercept(**KWARGS); fails+=1; print_red(f'Fail. HEADER_OR_FULL_SUPOBJ as full did not except when not enough columns.')
except: print_green(f'Pass.')


print_green(f'\nTesting HEADER_OR_FULL_SUPOBJ as full excepts if too many columns...')
KWARGS = deepcopy(BASE_KWARGS)
KWARGS['HEADER_OR_FULL_SUPOBJ'] = np.insert(KWARGS['HEADER_OR_FULL_SUPOBJ'], 0, KWARGS['HEADER_OR_FULL_SUPOBJ'][:,0], axis=1)
try: TestClass1.insert_intercept(**KWARGS); fails+=1; print_red(f'Fail. HEADER_OR_FULL_SUPOBJ as full did not except when too many columns.')
except: print_green(f'Pass.')


print_green(f'\nTesting HEADER_OR_FULL_SUPOBJ as full excepts if SUPOBJ_INSERT_VECTOR as header is PASSED...')
KWARGS = deepcopy(BASE_KWARGS)
KWARGS["SUPOBJ_INSERT_VECTOR"] = [['TEST INTERCEPT']]
try:
    TestClass1.insert_intercept(**KWARGS)
    fails+=1
    print_red(f'Fail. HEADER_OR_FULL_SUPOBJ did not except when passed SUPOBJ_INSERT_VECTOR as header to a full HEADER_OR_FULL_SUPOBJ.')
except: print_green(f'Pass.')


print_green(f'\nTesting excepts when simultaneously passed SUPOBJ_INSERT_VECTOR, header_string, datatype_string...')
KWARGS = deepcopy(BASE_KWARGS)
KWARGS["SUPOBJ_INSERT_VECTOR"] = ['TEST INTERCEPT', 'INT', 'INT', [], 0, 'N', 0, 0, '']
KWARGS["header_string"] = 'TEST INTERCEPT'
KWARGS["datatype_string"] = 'INT'
try:
    TestClass1.insert_intercept(**KWARGS)
    fails+=1
    print_red(f'Fail. Did not except when simultaneously passed SUPOBJ_INSERT_VECTOR, header_string, datatype_string.')
except: print_green(f'Pass.')


print_green(f'\nTesting excepts when none of SUPOBJ_INSERT_VECTOR, header_string, or datatype_string are passed ...')
KWARGS = deepcopy(BASE_KWARGS)
try:
    TestClass1.insert_intercept(**KWARGS)
    fails+=1
    print_red(f'Fail. Did not except when none of SUPOBJ_INSERT_VECTOR, header_string, datatype_string were passed.')
except: print_green(f'Pass.')


print_green(f'\nTesting excepts when passed only header_string for full HEADER_OR_FULL_SUPOBJ...')
KWARGS = deepcopy(BASE_KWARGS)
KWARGS["header_string"] = 'TEST INTERCEPT'
try: TestClass1.insert_intercept(**KWARGS); fails+=1; print_red(f'Fail. Did not except when only header_string passed for full HEADER_OR_FULL_SUPOBJ.')
except: print_green(f'Pass.')


print_green(f'\nTesting excepts when only passed datatype_string for full HEADER_OR_FULL_SUPOBJ...')
KWARGS = deepcopy(BASE_KWARGS)
KWARGS["datatype_string"] = 'INT'
try: TestClass1.insert_intercept(**KWARGS); fails+=1; print_red(f'Fail. Did not except when only datatype_string passed for full HEADER_OR_FULL_SUPOBJ.')
except: print_green(f'Pass.')


print_green(f'\nTesting excepts when header_string passed as a non-string for full HEADER_OR_FULL_SUPOBJ...')
KWARGS = deepcopy(BASE_KWARGS)
KWARGS["header_string"] =['INT']
KWARGS["datatype_string"] = 'INT'
try: TestClass1.insert_intercept(**KWARGS); fails+=1; print_red(f'Fail. Did not except when passed header_string as a non-string for full HEADER_OR_FULL_SUPOBJ.')
except: print_green(f'Pass.')


print_green(f'\nTesting excepts when datatype_string passed as a non-string for full HEADER_OR_FULL_SUPOBJ...')
KWARGS = deepcopy(BASE_KWARGS)
KWARGS["header_string"] = 'INT'
KWARGS["datatype_string"] = ['INT']
try: TestClass1.insert_intercept(**KWARGS); fails+=1; print_red(f'Fail. Did not except when passed datatype_string as a non-string for full HEADER_OR_FULL_SUPOBJ.')
except: print_green(f'Pass.')


print_green(f'\nTesting excepts when SUPOBJ_INSERT_VECTOR is too short for full HEADER_OR_FULL_SUPOBJ...')
KWARGS = deepcopy(BASE_KWARGS)
KWARGS["SUPOBJ_INSERT_VECTOR"] = ['TEST INTERCEPT', 'INT', 'INT', [], 0, 'N', 0, 0]
try: TestClass1.insert_intercept(**KWARGS); fails+=1; print_red(f'Fail. Did not except when SUPOBJ_INSERT_VECTOR is too short for full HEADER_OR_FULL_SUPOBJ.')
except: print_green(f'Pass.')

print_green(f'\nTesting excepts when SUPOBJ_INSERT_VECTOR is too long for full HEADER_OR_FULL_SUPOBJ...')
KWARGS = deepcopy(BASE_KWARGS)
KWARGS["SUPOBJ_INSERT_VECTOR"] = ['TEST INTERCEPT', 'INT', 'INT', [], 0, 'N', 0, 0, '', '']
try: TestClass1.insert_intercept(**KWARGS); fails+=1; print_red(f'Fail. Did not except when SUPOBJ_INSERT_VECTOR is too long for full HEADER_OR_FULL_SUPOBJ.')
except: print_green(f'Pass.')

# END HEADER_OR_FULL_SUPOBJ, SUPOBJ_INSERT_VECTOR, datatype_string, & header_string #################################


# CONTEXT HANDLED EXCLUSIVELY BY insert_column() #############################################################################
print_green(f'\nTesing CONTEXT other than None or [] excepts...')
KWARGS = deepcopy(BASE_KWARGS); KWARGS['CONTEXT'] = 'DUM'
try: TestClass1.insert_intercept(**KWARGS); print_red(f'Fail. Did not except when CONTEXT other than None or [] was passed.')
except: print_green(f'Pass.')
# END CONTEXT HANDLED EXCLUSIVELY BY insert_column() #########################################################################


if fails>0: print_red(f'\nFinished {test_name}. {fails} tests failed.')
else: print_green(f'\nFinished {test_name}. All tests passed.')
# END TEST DATA VALIDATION EXCEPTIONS ########################################################################################
##########################################################################################################################









# TEST FOR ACCURACY
print(f'\nStart tests for accuracy...')


_rows, _cols = 6, 4


MASTER_BYPASS_VALIDATION = [True, False]
MASTER_DATA_GIVEN_ORIENTATION = ['ROW', 'COLUMN']
MASTER_DATA_GIVEN_FORMAT = ['ARRAY', 'SPARSE_DICT']
MASTER_DATA_RETURN_FORMAT = ['ARRAY', 'SPARSE_DICT']
MASTER_DATA_RETURN_ORIENTATION = ['ROW', 'COLUMN']
MASTER_COL_IDX = [0, _cols//2, _cols]
MASTER_COLUMN_OR_VALUE_TO_INSERT_AND_INSERT_ORIENTATION = [
                                                            ('AS COLUMN', 'COLUMN'),
                                                            ('AS COLUMN', 'ROW'),
                                                            ('AS VALUE', None)
                                                           ]
MASTER_INSERT_ORIENTATION = ['ROW', 'COLUMN']
MASTER_SUPOBJ_FORMAT_AND_INSERT_INFO = [
                                            (None, None),
                                            ('AS HEADER', 'AS VECTOR'),
                                            ('AS HEADER', 'AS VALUES'),
                                            ('AS FULL', 'AS VECTOR'),
                                            ('AS FULL', 'AS VALUES')
                                        ]
MASTER_CONTEXT_PASSED = [True, False]

total_trials = np.product(list(map(len, (MASTER_BYPASS_VALIDATION, MASTER_DATA_GIVEN_FORMAT, MASTER_DATA_GIVEN_ORIENTATION,
                MASTER_DATA_RETURN_FORMAT, MASTER_DATA_RETURN_ORIENTATION, MASTER_COL_IDX,
                MASTER_COLUMN_OR_VALUE_TO_INSERT_AND_INSERT_ORIENTATION, MASTER_SUPOBJ_FORMAT_AND_INSERT_INFO, MASTER_CONTEXT_PASSED
))))

PRIMAL_DATA = np.random.randint(0,10,(_rows, _cols))
PRIMAL_FULL_SUPOBJ = bfso.BuildFullSupportObject(OBJECT=PRIMAL_DATA, object_given_orientation="ROW",
                          OBJECT_HEADER=th.test_header(_cols), SUPPORT_OBJECT=None, columns=_cols, quick_vdtypes=False,
                          MODIFIED_DATATYPES=None, print_notes=False, prompt_to_override=False, bypass_validation=True,
                          calling_module=calling_module, calling_fxn=test_name
).SUPPORT_OBJECT
PRIMAL_HEADER_SUPOBJ = th.test_header(_cols)



ctr = 0
for bypass_validation in MASTER_BYPASS_VALIDATION:
    for data_given_orientation in MASTER_DATA_GIVEN_ORIENTATION:
        if data_given_orientation=='COLUMN': BASE_DATA = PRIMAL_DATA.copy().transpose()
        else: BASE_DATA = PRIMAL_DATA.copy()
        for data_given_format in MASTER_DATA_GIVEN_FORMAT:
            if data_given_format=='SPARSE_DICT': BASE_DATA = sd.zip_list_as_py_int(BASE_DATA)
            for data_return_format in MASTER_DATA_RETURN_FORMAT:
                for data_return_orientation in MASTER_DATA_RETURN_ORIENTATION:
                    for col_idx in MASTER_COL_IDX:
                        for (column_or_value_to_insert, insert_orientation) in MASTER_COLUMN_OR_VALUE_TO_INSERT_AND_INSERT_ORIENTATION:
                            for (supobj_format, supobj_insert_info) in MASTER_SUPOBJ_FORMAT_AND_INSERT_INFO:
                                for context_passed in MASTER_CONTEXT_PASSED:
                                    ctr += 1
                                    print(f'\nRunning trial {ctr} of {total_trials}...')

                                    GIVEN_DATA = deepcopy(BASE_DATA) if isinstance(BASE_DATA, dict) else BASE_DATA.copy()

                                    # column_or_value_to_insert, COLUMN_OR_VALUE_TO_INSERT, ['AS COLUMN', 'AS VALUE']
                                    if column_or_value_to_insert == 'AS COLUMN':
                                        COLUMN_OR_VALUE_TO_INSERT = np.ones(_rows)
                                        if insert_orientation == 'COLUMN': pass
                                        elif insert_orientation == 'ROW':
                                            if len(COLUMN_OR_VALUE_TO_INSERT.shape)==1:
                                                COLUMN_OR_VALUE_TO_INSERT = COLUMN_OR_VALUE_TO_INSERT.reshape((1,-1))
                                            COLUMN_OR_VALUE_TO_INSERT = COLUMN_OR_VALUE_TO_INSERT.transpose()
                                        else: raise Exception(f'INVALID insert_orientation {insert_orientation}')
                                    elif column_or_value_to_insert=='AS VALUE':
                                        COLUMN_OR_VALUE_TO_INSERT = 2
                                    else: raise Exception(f'INVALID column_or_value_to_insert {column_or_value_to_insert}')

                                    # supobj_format, HEADER_OR_FULL_SUPOBJ, [None, 'AS HEADER', 'AS FULL']
                                    if supobj_format is None: HEADER_OR_FULL_SUPOBJ = None
                                    elif supobj_format == 'AS HEADER': HEADER_OR_FULL_SUPOBJ = PRIMAL_HEADER_SUPOBJ.copy()
                                    elif supobj_format == 'AS FULL': HEADER_OR_FULL_SUPOBJ = PRIMAL_FULL_SUPOBJ.copy()
                                    else: raise Exception(f'INVALID supobj_format "{supobj_format}"')

                                    # supobj_insert_info, (insert_orientation, SUPOBJ_INSERT_VECTOR, ['ROW', 'COLUMN']), datatype_string, header_string, ['AS VECTOR', 'AS VALUES']
                                    if supobj_insert_info is None:
                                        SUPOBJ_INSERT_VECTOR = None
                                        datatype_string = None
                                        header_string = None
                                    elif supobj_insert_info == 'AS VECTOR':
                                        if supobj_format == 'AS HEADER': SUPOBJ_INSERT_VECTOR = ['INTERCEPT']
                                        elif supobj_format == 'AS FULL': SUPOBJ_INSERT_VECTOR = ['INTERCEPT', 'INT', 'INT', [], 0, 'N', 0, 0, '']
                                        datatype_string = None
                                        header_string = None
                                    elif supobj_insert_info == 'AS VALUES':
                                        SUPOBJ_INSERT_VECTOR = None
                                        if supobj_format == 'AS HEADER':
                                            datatype_string = None
                                            header_string = 'INTERCEPT'
                                        elif supobj_format == 'AS FULL':
                                            datatype_string = 'INT'
                                            header_string = 'INTERCEPT'
                                    else: raise Exception(f'INVALID supobj_insert_info "{supobj_insert_info}"')


                                    # context_passed, CONTEXT, [True, False]
                                    CONTEXT = ['Dummy placeholder.'] if context_passed is True else None

                                    TestClass1 = mlo.MLObject(
                                                              GIVEN_DATA,
                                                              data_given_orientation,
                                                              name="GIVEN DATA",
                                                              return_orientation=data_return_orientation,
                                                              return_format=data_return_format,
                                                              bypass_validation=bypass_validation,
                                                              calling_module=calling_module,
                                                              calling_fxn='accuracy tests'
                                    )

                                    # ANSWER KEY #######################################################################################################

                                    ANSWER_DATA = PRIMAL_DATA.copy()
                                    if data_return_orientation == 'COLUMN': ANSWER_DATA = ANSWER_DATA.transpose()
                                    ANSWER_DATA = np.insert(ANSWER_DATA, col_idx, 1 if column_or_value_to_insert=='AS COLUMN' else 2,
                                                                 axis=0 if data_return_orientation == 'COLUMN' else 1)

                                    # DONT ZIP ANSWER TO SPARSE DICT, RETURNED DATA WILL BE UNZIPPED TO NP ARRAY IF SPARSE DICT THEN TESTED AGAINST KEY

                                    if supobj_format=='AS FULL':
                                        ANSWER_SUPOBJ = np.insert(PRIMAL_FULL_SUPOBJ.copy(), col_idx, ['INTERCEPT','INT','INT',[],0,'N',0,0,''], axis=1)
                                    elif supobj_format=='AS HEADER':
                                        ANSWER_SUPOBJ = np.insert(PRIMAL_HEADER_SUPOBJ.copy(), col_idx, 'INTERCEPT', axis=1)
                                    elif supobj_format is None: ANSWER_SUPOBJ = None
                                    else: raise Exception(f'INVALID supobj_format "{supobj_format}"')

                                    if context_passed:
                                        if not HEADER_OR_FULL_SUPOBJ is None:
                                            ANSWER_CONTEXT = [f'Dummy placeholder.',
                                                                f'Inserted INTERCEPT INTO GIVEN DATA in the {col_idx} index position.']
                                        elif HEADER_OR_FULL_SUPOBJ is None:
                                            ANSWER_CONTEXT = [f'Dummy placeholder.',
                                                                f'Inserted unnamed column INTO GIVEN DATA in the {col_idx} index position.']
                                    elif not context_passed:
                                        if not HEADER_OR_FULL_SUPOBJ is None:
                                            ANSWER_CONTEXT = [f'Inserted INTERCEPT INTO GIVEN DATA in the {col_idx} index position.']
                                        elif HEADER_OR_FULL_SUPOBJ is None:
                                            ANSWER_CONTEXT = [f'Inserted unnamed column INTO GIVEN DATA in the {col_idx} index position.']
                                    # END ANSWER KEY #######################################################################################################


                                    ACT_DATA = TestClass1.insert_intercept(
                                                                           col_idx,
                                                                           COLUMN_OR_VALUE_TO_INSERT,
                                                                           insert_orientation=insert_orientation,
                                                                           HEADER_OR_FULL_SUPOBJ=HEADER_OR_FULL_SUPOBJ,
                                                                           SUPOBJ_INSERT_VECTOR=SUPOBJ_INSERT_VECTOR,
                                                                           datatype_string=datatype_string,
                                                                           header_string=header_string,
                                                                           CONTEXT=CONTEXT
                                    )
                                    if data_return_format=='SPARSE_DICT': ACT_DATA = sd.unzip_to_ndarray_float64(ACT_DATA)[0].astype(np.int32)
                                    ACT_SUPOBJ = TestClass1.HEADER_OR_FULL_SUPOBJ
                                    ACT_CONTEXT = TestClass1.CONTEXT

                                    trial_description = \
                                        f'\nbypass_validation = {bypass_validation}' \
                                        f'\ndata_given_format = {data_given_format}' \
                                        f'\ndata_given_orientation = {data_given_orientation}' \
                                        f'\ndata_return_format = {data_return_format}' \
                                        f'\ndata_return_orientation = {data_return_orientation}' \
                                        f'\ncol_idx = {col_idx}' \
                                        f'\ncolumn_or_value_to_insert = {column_or_value_to_insert}' \
                                        f'\ninsert_orientation = {insert_orientation}' \
                                        f'\nsupobj_format = {supobj_format}' \
                                        f'\nsupobj_insert_info = {supobj_insert_info}' \
                                        f'\ncontext_passed = {context_passed}'

                                    # TEST ACT & ANSWER DATA CONGRUENCE
                                    if not np.array_equiv(ANSWER_DATA, ACT_DATA):
                                        print_red(f'\nTrial description:')
                                        print_red(trial_description)
                                        print_red(f'\nANSWER DATA AND ACTUAL DATA ARE NOT CONGRUENT')
                                        print_red(f'ANSWER DATA = ')
                                        print_red(ANSWER_DATA)
                                        print()
                                        print_red(f'ACTUAL DATA = ')
                                        print_red(ACT_DATA)
                                        wls.winlinsound(222, 3)
                                        raise Exception(f'FAIL.')

                                    # TEST ACT & ANSWER SUPOBJ CONGRUENCE
                                    if not np.array_equiv(ANSWER_SUPOBJ, ACT_SUPOBJ):
                                        print_red(f'\nTrial description:')
                                        print_red(trial_description)
                                        print_red(f'\nANSWER SUPOBJ AND ACTUAL SUPOBJ ARE NOT CONGRUENT')
                                        print_red(f'ANSWER SUPOBJ = ')
                                        print_red(ANSWER_SUPOBJ)
                                        print()
                                        print_red(f'ACTUAL SUPOBJ = ')
                                        print_red(ACT_SUPOBJ)
                                        wls.winlinsound(222, 3)
                                        raise Exception(f'FAIL.')

                                    # TEST ACT & ANSWER CONTEXT CONGRUENCE
                                    if not np.array_equiv(ANSWER_CONTEXT, ACT_CONTEXT):
                                        print_red(f'\nTrial description:')
                                        print_red(trial_description)
                                        print_red(f'\nANSWER CONTEXT AND ACTUAL CONTEXT ARE NOT CONGRUENT')
                                        print_red(f'ANSWER CONTEXT = ')
                                        print_red(ANSWER_CONTEXT)
                                        print()
                                        print_red(f'ACTUAL CONTEXT = ')
                                        print_red(ACT_CONTEXT)
                                        wls.winlinsound(222, 3)
                                        raise Exception(f'FAIL.')




# IF REACH THIS POINT, PERMUTATION TESTS PASSED, SO IF fails==0, ALL TESTS PASSED
if fails==0:
    print_green(f'*** ALL TESTS PASSED ***')
    for _ in range(3): wls.winlinsound(888, 1); time.sleep(0.5)
else: print_red(f'*** PERMUTATION TESTS PASSED, BUT {fails} VALIDATION TESTS FAILED ***'); wls.winlinsound(222, 3)


# col_idx
# HANDLED EXCLUSIVELY BY insert_column()

# COLUMN_OR_VALUE_TO_INSERT & insert_orientation ################################################################
# IF VALUE IS PASSED, CONVERT TO LIST (insert_column(COLUMN_TO_INSERT) MUST TAKE LIST)
#       - EVERYTHING ELSE ABOUT THE LIST IS VALIDATED IN insert_column()
#       - insert_orientation MUST BE PASSED... insert_column() REQUIRES IT
# IF LIST IS PASSED
#       - insert_orientation MUST BE PASSED (BUT NOT IF A VALUE)... insert_column() REQUIRES IT
#       - VERIFY IS CONSTANT
#       - EVERYTHING ELSE ABOUT THE LIST IS VALIDATED IN insert_column()
# END COLUMN_OR_VALUE_TO_INSERT & insert_orientation ############################################################

# HEADER_OR_FULL_SUPOBJ, SUPOBJ_INSERT_VECTOR, datatype_string, & header_string #################################
# IF HEADER_OR_FULL_SUPOBJ IS NOT PASSED
#       - SUPOBJ_INSERT_VECTOR, datatype_string, AND header_string CANNOT BE PASSED
# IF HEADER_OR_FULL_SUPOBJ IS PASSED
#       - ROWS IN HEADER_OR_FULL_SUPOBJ MUST MATCH UP TO HEADER ONLY OR A FULL SUPOBJ
#       - (SUPOBJ_INSERT_VECTOR) OR (datatype_string) or (datatype_string & header_string) CANNOT BE None
#       - IF SUPOBJ_INSERT_VECTOR IS PASSED
#           - datatype_str AND header_str CANNOT BE PASSED
#           - SUPOBJ_INSERT_VECTOR MUST BE EQUALLY SIZED WITH HEADER_OR_FULL_SUPOBJ (FULL WITH FULL, HDR WITH HDR)
#       - IF SUPOBJ_INSERT_VECTOR IS NOT PASSED, header_string OR (header_string & datatype_string) MUST BE PASSED'
#       - SUPOBJ_INSERT_VECTOR MUST BE CONSTRUCTED FROM datatype_string & header_string (insert_column() REQUIRES A VECTOR)
#           - IF HEADER_OR_FULL_SUPOBJ IS HEADER ONLY, header_string MUST BE PASSED, IGNORE datatype_string
#           - IF HEADER_OR_FULL_SUPOBJ IS FULL, header_string & datatype_string MUST BE PASSED
#        - SUPOBJ_INSERT_VECTOR IS PASSED TO insert_column(SUPOBJ_INSERT) AND IS VALIDATED THERE

# CONTEXT
# HANDLED EXCLUSIVELY BY insert_column()













