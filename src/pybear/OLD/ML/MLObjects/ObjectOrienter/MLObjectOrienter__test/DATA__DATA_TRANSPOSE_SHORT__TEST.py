import numpy as np
import sparse_dict as sd
import sys
from debug import get_module_name as gmn
from MLObjects.ObjectOrienter import MLObjectOrienter as mloo


# A MODULE TO VALIDATE FUNCTIONALITY OF MLObjectOrienter W/ DATA &/OR DATA_TRANSPOSE AS INPUTS, AND OUTPUTS OF
# DATA, DATA_TRANSPOSE, XTX, AND XTX_INV
# VERIFIED TEST CODE WORKS AND MLObjectOrienter PASSES FOR DATA, DATA_TRANSPOSE, XTX, & XTX_INV OUTPUTS 12/18/22
# INPUTS OF XTX, XTX_INV, TARGET, TARGET_TRANSPOSE, TARGET_AS_LIST TESTED IN OTHER MODULES TO MANAGE COMPLEXITY


def _error(words):
    raise Exception(f'\n*** {words} ***\n')




calling_module = gmn.get_module_name(str(sys.modules[__name__]))
calling_fxn = 'guard_test'


RAW_DATA = [[0 ,1 ,2] ,[3 ,4 ,5] ,[6 ,7 ,8], [9, 10, 11]]
DATA_NP_TEMPLATE = np.array(RAW_DATA)
DATA_SD_TEMPLATE = sd.zip_list_as_py_float(DATA_NP_TEMPLATE)


MASTER_BYPASS_VALIDATION = [True, False]
MASTER_DATA_OBJECTS = [
                        DATA_NP_TEMPLATE,
                        DATA_NP_TEMPLATE,
                        DATA_NP_TEMPLATE,
                        DATA_NP_TEMPLATE,
                        DATA_NP_TEMPLATE,
                        DATA_NP_TEMPLATE,
                        None,
                        None,
                        None,
                        None,
                        DATA_SD_TEMPLATE,
                        DATA_SD_TEMPLATE,
                        DATA_SD_TEMPLATE,
                        DATA_SD_TEMPLATE,
                        DATA_SD_TEMPLATE,
                        DATA_SD_TEMPLATE,
                        None,
                        None,
                        None,
                        None
                        ]

MASTER_DATA_GIVEN_ORIENTATION = ['ROW', 'ROW', 'ROW', 'COLUMN', 'COLUMN', 'COLUMN', None, None, None, None,
                                 'ROW', 'ROW', 'ROW', 'COLUMN', 'COLUMN', 'COLUMN', None, None, None, None]

MASTER_DATA_TRANSPOSE_OBJECTS = [
                                 DATA_NP_TEMPLATE.transpose(),
                                 sd.sparse_transpose(DATA_SD_TEMPLATE),
                                 None,
                                 DATA_NP_TEMPLATE.transpose(),
                                 sd.sparse_transpose(DATA_SD_TEMPLATE),
                                 None,

                                 DATA_NP_TEMPLATE.transpose(),
                                 sd.sparse_transpose(DATA_SD_TEMPLATE),
                                 DATA_NP_TEMPLATE.transpose(),
                                 sd.sparse_transpose(DATA_SD_TEMPLATE),

                                 DATA_NP_TEMPLATE.transpose(),
                                 sd.sparse_transpose(DATA_SD_TEMPLATE),
                                 None,
                                 DATA_NP_TEMPLATE.transpose(),
                                 sd.sparse_transpose(DATA_SD_TEMPLATE),
                                 None,

                                 DATA_NP_TEMPLATE,
                                 DATA_SD_TEMPLATE,
                                 DATA_NP_TEMPLATE,
                                 DATA_SD_TEMPLATE,
                                 ]

MASTER_DATA_TRANSPOSE_GIVEN_ORIENTATION = ['ROW', 'ROW', None, 'COLUMN', 'COLUMN', None, 'COLUMN', 'COLUMN', 'ROW', 'ROW',
                                           'ROW', 'ROW', None, 'COLUMN', 'COLUMN', None, 'COLUMN', 'COLUMN', 'ROW', 'ROW']

MASTER_DATA_RETURN_ORIENTATION = ['ROW', 'COLUMN', 'AS_GIVEN']
MASTER_DATA_RETURN_FORMAT = ['ARRAY', 'SPARSE_DICT', 'AS_GIVEN']
MASTER_DATA_TRANSPOSE_RETURN_ORIENTATION = ['ROW', 'COLUMN', 'AS_GIVEN']
MASTER_DATA_TRANSPOSE_RETURN_FORMAT = ['ARRAY', 'SPARSE_DICT', 'AS_GIVEN']
MASTER_RETURN_OBJECTS = [['DATA'], ['DATA_TRANSPOSE'], ['DATA', 'DATA_TRANSPOSE'], []]

total_trials = np.product(list(map(len, [MASTER_BYPASS_VALIDATION, MASTER_DATA_OBJECTS, MASTER_DATA_RETURN_ORIENTATION,
    MASTER_DATA_RETURN_FORMAT, MASTER_DATA_TRANSPOSE_RETURN_ORIENTATION, MASTER_DATA_TRANSPOSE_RETURN_FORMAT, MASTER_RETURN_OBJECTS])))


ctr = 0
for bypass_validation in MASTER_BYPASS_VALIDATION:
    for DATA, DATA_TRANSPOSE, data_given_orientation, data_transpose_given_orientation in \
            zip(MASTER_DATA_OBJECTS, MASTER_DATA_TRANSPOSE_OBJECTS, MASTER_DATA_GIVEN_ORIENTATION, MASTER_DATA_TRANSPOSE_GIVEN_ORIENTATION):
        for data_return_orientation in MASTER_DATA_RETURN_ORIENTATION:
            for data_return_format in MASTER_DATA_RETURN_FORMAT:
                for data_transpose_return_orientation in MASTER_DATA_TRANSPOSE_RETURN_ORIENTATION:
                    for data_transpose_return_format in MASTER_DATA_TRANSPOSE_RETURN_FORMAT:
                        for DUM_RETURN_OBJECTS in MASTER_RETURN_OBJECTS:

                            if DATA is None and DATA_TRANSPOSE is None and ('DATA' in DUM_RETURN_OBJECTS or 'DATA_TRANSPOSE' in DUM_RETURN_OBJECTS):
                                total_trials -= 1
                                continue

                            ctr += 1
                            if ctr % 2000 == 0:
                                print(f'*'*90)
                                print(f'Running trial {ctr} of at most {total_trials}...')

                                print(f'DATA is {"not given" if DATA is None else "given"}, '
                                      f'DATA_TRANSPOSE is {"not given" if DATA_TRANSPOSE is None else "given"}.')

                                print(f'DUM_RETURN_OBJECTS = ')
                                print(DUM_RETURN_OBJECTS)

                            exp_calling_module = calling_module
                            exp_calling_fxn = calling_fxn


                            ############################################################################################################
                            if (DATA is None and DATA_TRANSPOSE is None) or \
                                    ('DATA' not in DUM_RETURN_OBJECTS and 'DATA_TRANSPOSE' not in DUM_RETURN_OBJECTS):
                                EXP_DATA = None
                                exp_data_current_orientation = None
                                exp_data_return_orientation = None
                                exp_data_current_format = None
                                exp_data_return_format = None
                                EXP_DATA_TRANSPOSE = None
                                exp_data_transpose_current_orientation = None
                                exp_data_transpose_return_orientation = None
                                exp_data_transpose_current_format = None
                                exp_data_transpose_return_format = None
                            ############################################################################################################

                            ############################################################################################################
                            elif not DATA is None and DATA_TRANSPOSE is None:

                                # MUST BUILD DATA, EVEN IF NOT RETURNED, TO BUILD DATA_TRANSPOSE

                                exp_data_given_format = 'SPARSE_DICT' if isinstance(DATA, dict) else 'ARRAY' if isinstance(DATA, np.ndarray) else None

                                if data_return_format != 'AS_GIVEN':
                                    exp_data_current_format = data_return_format
                                    exp_data_return_format = data_return_format
                                elif data_return_format == 'AS_GIVEN':
                                    exp_data_current_format = exp_data_given_format
                                    exp_data_return_format = exp_data_given_format

                                if exp_data_given_format == 'ARRAY' and exp_data_return_format == 'SPARSE_DICT':
                                    EXP_DATA = sd.zip_list_as_py_float(DATA)
                                elif exp_data_given_format == 'SPARSE_DICT' and exp_data_return_format == 'ARRAY':
                                    EXP_DATA = sd.unzip_to_ndarray_float64(DATA)[0]
                                else:    # return format == given format so no change
                                    EXP_DATA = DATA

                                # EXP DATA IS CORRECT RETURN FORMAT BUT NOT CORRECT ORIENTATION YET

                                exp_data_given_orientation = data_given_orientation

                                if data_return_orientation != 'AS_GIVEN':
                                    exp_data_current_orientation = data_return_orientation
                                    exp_data_return_orientation = data_return_orientation
                                elif data_return_orientation == 'AS_GIVEN':
                                    exp_data_current_orientation = data_given_orientation
                                    exp_data_return_orientation = data_given_orientation

                                if exp_data_return_orientation == data_given_orientation:
                                    pass #EXP_DATA = EXP_DATA FOR BOTH ARRAY AND SPARSE_DICT
                                elif exp_data_return_orientation != data_given_orientation:
                                    if exp_data_return_format == 'ARRAY': EXP_DATA = EXP_DATA.transpose()
                                    elif exp_data_return_format == 'SPARSE_DICT': EXP_DATA = sd.sparse_transpose(EXP_DATA)

                                # FINAL RETURN FORMAT AND ORIENTATION OF EXP_DATA ARE FINISHED

                                # IF NOT NEEDED, DONT BOTHER TO MAKE
                                if 'DATA_TRANSPOSE' not in DUM_RETURN_OBJECTS:
                                    EXP_DATA_TRANSPOSE = None

                                    exp_data_transpose_given_format = None
                                    exp_data_transpose_given_orientation = None

                                    # "RETURN" THINGS THINKING SHOULD BE None
                                    exp_data_transpose_current_format = None
                                    exp_data_transpose_return_format = None
                                    exp_data_transpose_current_orientation = None
                                    exp_data_transpose_return_orientation = None
                                else:
                                    # BECAUSE DATA_T WAS NOT GIVEN AND WILL COME FROM COPY OF DATA
                                    exp_data_transpose_given_format = None
                                    exp_data_transpose_given_orientation = None

                                    if data_transpose_return_format != 'AS_GIVEN':
                                        exp_data_transpose_current_format = data_transpose_return_format
                                        exp_data_transpose_return_format = data_transpose_return_format
                                    elif data_transpose_return_format == 'AS_GIVEN':
                                        exp_data_transpose_current_format = exp_data_given_format
                                        exp_data_transpose_return_format = exp_data_given_format

                                    if exp_data_return_format == 'ARRAY' and exp_data_transpose_return_format == 'SPARSE_DICT':
                                        EXP_DATA_TRANSPOSE = sd.zip_list_as_py_float(EXP_DATA.transpose())
                                    elif exp_data_return_format == 'SPARSE_DICT' and exp_data_transpose_return_format == 'ARRAY':
                                        EXP_DATA_TRANSPOSE = sd.unzip_to_ndarray_float64(EXP_DATA)[0].transpose()
                                    elif exp_data_return_format == exp_data_transpose_return_format:
                                        # STILL HAVE TO TRANSPOSE DATA OBJECT TO DATA_TRANSPOSE
                                        if exp_data_return_format == 'ARRAY':  EXP_DATA_TRANSPOSE = EXP_DATA.transpose()
                                        elif exp_data_return_format == 'SPARSE_DICT': EXP_DATA_TRANSPOSE = sd.sparse_transpose(EXP_DATA)

                                    # AT THIS POINT HAVE DATA IN FINAL RETURN FORMAT & ORIENTATION, DATA_TRANSPOSE IN RETURN FORMAT

                                    # BECAUSE DATA_T WAS NOT GIVEN AND WILL COME FROM COPY OF DATA
                                    exp_data_transpose_given_orientation = None

                                    if data_transpose_return_orientation != 'AS_GIVEN':
                                        exp_data_transpose_current_orientation = data_transpose_return_orientation
                                        exp_data_transpose_return_orientation = data_transpose_return_orientation
                                    elif data_transpose_return_orientation == 'AS_GIVEN':
                                        exp_data_transpose_current_orientation = exp_data_given_orientation
                                        exp_data_transpose_return_orientation = exp_data_given_orientation

                                    if exp_data_transpose_return_orientation == exp_data_return_orientation:
                                        pass
                                    elif exp_data_transpose_return_orientation != exp_data_return_orientation:
                                        if exp_data_transpose_return_format == 'ARRAY':
                                            EXP_DATA_TRANSPOSE = EXP_DATA_TRANSPOSE.transpose()
                                        elif exp_data_transpose_return_format == 'SPARSE_DICT':
                                            EXP_DATA_TRANSPOSE = sd.sparse_transpose(EXP_DATA_TRANSPOSE)

                                    # NOW HAVE DATA & DATA_TRANSPOSE IN CORRECT RETURN FORMAT AND ORIENTATION

                            ############################################################################################################

                            ############################################################################################################
                            elif DATA is None and not DATA_TRANSPOSE is None:
                                # MUST BUILD DATA_TRANSPOSE, EVEN IF NOT RETURNED, TO BUILD DATA
                                exp_data_transpose_given_format = 'SPARSE_DICT' if isinstance(DATA_TRANSPOSE, dict) else 'ARRAY' if isinstance(DATA_TRANSPOSE, np.ndarray) else None

                                if data_transpose_return_format != 'AS_GIVEN':
                                    exp_data_transpose_current_format = data_transpose_return_format
                                    exp_data_transpose_return_format = data_transpose_return_format
                                elif data_transpose_return_format == 'AS_GIVEN':
                                    exp_data_transpose_current_format = exp_data_transpose_given_format
                                    exp_data_transpose_return_format = exp_data_transpose_given_format

                                if exp_data_transpose_given_format == 'ARRAY' and exp_data_transpose_return_format == 'SPARSE_DICT':
                                    EXP_DATA_TRANSPOSE = sd.zip_list_as_py_float(DATA_TRANSPOSE)
                                elif exp_data_transpose_given_format == 'SPARSE_DICT' and exp_data_transpose_return_format == 'ARRAY':
                                    EXP_DATA_TRANSPOSE = sd.unzip_to_ndarray_float64(DATA_TRANSPOSE)[0]
                                else:    # return format == given format so no change
                                    EXP_DATA_TRANSPOSE = DATA_TRANSPOSE   # FOR WHEN STAYS ARRAY OR STAYS SPARSE_DICT

                                # NOW HAVE DATA_TRANSPOSE IN CORRECT RETURN FORMAT, BUT NOT CORRECT ORIENTATION YET

                                exp_data_transpose_given_orientation = data_transpose_given_orientation

                                if data_transpose_return_orientation != 'AS_GIVEN':
                                    exp_data_transpose_current_orientation = data_transpose_return_orientation
                                    exp_data_transpose_return_orientation = data_transpose_return_orientation
                                elif data_transpose_return_orientation == 'AS_GIVEN':
                                    exp_data_transpose_current_orientation = data_transpose_given_orientation
                                    exp_data_transpose_return_orientation = data_transpose_given_orientation

                                if exp_data_transpose_return_orientation == exp_data_transpose_given_orientation:
                                    pass    # EXP_DATA_TRANSPOSE = EXP_DATA_TRANSPOSE FOR BOTH ARRAY AND SPARSE_DICT
                                elif exp_data_transpose_return_orientation != exp_data_transpose_given_orientation:
                                    if exp_data_transpose_return_format == 'ARRAY':
                                                            EXP_DATA_TRANSPOSE = EXP_DATA_TRANSPOSE.transpose()
                                    elif exp_data_transpose_return_format == 'SPARSE_DICT':
                                                            EXP_DATA_TRANSPOSE = sd.sparse_transpose(EXP_DATA_TRANSPOSE)

                                # NOW HAVE EXP_DATA_TRANSPOSE IN CORRECT RETURN FORMAT AND ORIENTATION

                                # IF NOT NEEDED, DONT BOTHER TO MAKE DATA
                                if 'DATA' not in DUM_RETURN_OBJECTS:
                                    EXP_DATA = None

                                    exp_data_given_format = None
                                    exp_data_given_orientation = None

                                    # "RETURN" THINGS THINKING SHOULD BE None
                                    exp_data_current_format = None
                                    exp_data_return_format = None
                                    exp_data_current_orientation = None
                                    exp_data_return_orientation = None

                                else:
                                    # BECAUSE DATA WAS NOT GIVEN AND WILL COME FROM COPY OF DATA_T
                                    exp_data_given_format = None

                                    if data_return_format != 'AS_GIVEN':
                                        exp_data_current_format = data_return_format
                                        exp_data_return_format = data_return_format
                                    elif data_return_format == 'AS_GIVEN':
                                        exp_data_current_format = exp_data_transpose_given_format
                                        exp_data_return_format = exp_data_transpose_given_format

                                    if exp_data_transpose_return_format == 'ARRAY' and exp_data_return_format == 'SPARSE_DICT':
                                        EXP_DATA = sd.zip_list_as_py_float(EXP_DATA_TRANSPOSE.transpose())
                                    elif exp_data_transpose_return_format == 'SPARSE_DICT' and exp_data_return_format == 'ARRAY':
                                        EXP_DATA = sd.unzip_to_ndarray_float64(EXP_DATA_TRANSPOSE)[0].transpose()
                                    elif exp_data_transpose_return_format == exp_data_return_format:
                                        # STILL HAVE TO TRANSPOSE DATA_TRANSPOSE OBJECT TO DATA
                                        if exp_data_transpose_return_format == 'ARRAY':  EXP_DATA = EXP_DATA_TRANSPOSE.transpose()
                                        elif exp_data_transpose_return_format == 'SPARSE_DICT': EXP_DATA = sd.sparse_transpose(EXP_DATA_TRANSPOSE)

                                    # NOW HAVE DATA_TRANSPOSE IN RETURN FORMAT & ORIENTATION, DATA IN CORRECT RETURN FORMAT

                                    # BECAUSE DATA WAS NOT GIVEN AND WILL COME FROM COPY OF DATA_T
                                    exp_data_given_orientation = None

                                    if data_return_orientation != 'AS_GIVEN':
                                        exp_data_current_orientation = data_return_orientation
                                        exp_data_return_orientation = data_return_orientation
                                    elif data_return_orientation == 'AS_GIVEN':
                                        exp_data_current_orientation = exp_data_transpose_given_orientation
                                        exp_data_return_orientation = exp_data_transpose_given_orientation

                                    if exp_data_return_orientation == exp_data_transpose_return_orientation:
                                        pass
                                    elif exp_data_return_orientation != exp_data_transpose_return_orientation:
                                        if exp_data_return_format == 'ARRAY': EXP_DATA = EXP_DATA.transpose()
                                        elif exp_data_return_format == 'SPARSE_DICT': EXP_DATA = sd.sparse_transpose(EXP_DATA)

                                # NOW HAVE BOTH DATA_TRANSPOSE & DATA IN CORRECT RETURN FORMAT & ORIENTATION

                            ############################################################################################################

                            ############################################################################################################
                            elif not DATA is None and not DATA_TRANSPOSE is None:

                                exp_data_given_format = 'SPARSE_DICT' if isinstance(DATA, dict) else 'ARRAY'

                                exp_data_given_orientation = data_given_orientation

                                if 'DATA' not in DUM_RETURN_OBJECTS:
                                    EXP_DATA = None
                                    exp_data_return_orientation = None
                                    exp_data_current_orientation = None
                                    exp_data_return_format = None
                                    exp_data_current_format = None


                                elif 'DATA' in DUM_RETURN_OBJECTS:

                                    if data_return_format != 'AS_GIVEN':
                                        exp_data_current_format = data_return_format
                                        exp_data_return_format = data_return_format
                                    elif data_return_format == 'AS_GIVEN':
                                        exp_data_current_format = exp_data_given_format
                                        exp_data_return_format = exp_data_given_format

                                    if exp_data_given_format == 'ARRAY' and exp_data_return_format == 'SPARSE_DICT':
                                        EXP_DATA = sd.zip_list_as_py_float(DATA)
                                    elif exp_data_given_format == 'SPARSE_DICT' and exp_data_return_format == 'ARRAY':
                                        EXP_DATA = sd.unzip_to_ndarray_float64(DATA)[0]
                                    else:   # return format == given format so no change
                                        EXP_DATA = DATA

                                    # NOW HAVE EXP_DATA IN CORRECT RETURN FORMAT

                                    if data_return_orientation != 'AS_GIVEN':
                                        exp_data_current_orientation = data_return_orientation
                                        exp_data_return_orientation = data_return_orientation
                                    elif data_return_orientation == 'AS_GIVEN':
                                        exp_data_current_orientation = data_given_orientation
                                        exp_data_return_orientation = data_given_orientation

                                    if exp_data_return_orientation == exp_data_given_orientation:
                                        pass  # EXP_DATA = EXP_DATA FOR BOTH ARRAY AND SPARSE_DICT
                                    elif exp_data_return_orientation != exp_data_given_orientation:
                                        if exp_data_return_format == 'ARRAY': EXP_DATA = EXP_DATA.transpose()
                                        elif exp_data_return_format == 'SPARSE_DICT': EXP_DATA = sd.sparse_transpose(EXP_DATA)

                                    # NOW HAVE EXP_DATA IN CORRECT RETURN FORMAT AND ORIENTATION

                                exp_data_transpose_given_format = 'SPARSE_DICT' if isinstance(DATA_TRANSPOSE, dict) else 'ARRAY' if isinstance(DATA_TRANSPOSE, np.ndarray) else None

                                exp_data_transpose_given_orientation = data_transpose_given_orientation

                                if 'DATA_TRANSPOSE' not in DUM_RETURN_OBJECTS:
                                    EXP_DATA_TRANSPOSE = None

                                    exp_data_transpose_return_format = None
                                    exp_data_transpose_current_format = None
                                    exp_data_transpose_return_orientation = None
                                    exp_data_transpose_current_orientation = None

                                elif 'DATA_TRANSPOSE' in DUM_RETURN_OBJECTS:
                                    if data_transpose_return_format != 'AS_GIVEN':
                                        exp_data_transpose_current_format = data_transpose_return_format
                                        exp_data_transpose_return_format = data_transpose_return_format
                                    elif data_transpose_return_format == 'AS_GIVEN':
                                        exp_data_transpose_current_format = exp_data_transpose_given_format
                                        exp_data_transpose_return_format = exp_data_transpose_given_format

                                    if exp_data_transpose_given_format == 'ARRAY' and exp_data_transpose_return_format == 'SPARSE_DICT':
                                        EXP_DATA_TRANSPOSE = sd.zip_list_as_py_float(DATA_TRANSPOSE)
                                    elif exp_data_transpose_given_format == 'SPARSE_DICT' and exp_data_transpose_return_format == 'ARRAY':
                                        EXP_DATA_TRANSPOSE = sd.unzip_to_ndarray_float64(DATA_TRANSPOSE)[0]
                                    else:    # return format == given format so no change
                                        EXP_DATA_TRANSPOSE = DATA_TRANSPOSE

                                    # NOW HAVE EXP_DATA_TRANSPOSE IN CORRECT RETURN FORMAT, & EXP_DATA IN RETURN FORMAT & ORIENTATION

                                    if data_transpose_return_orientation != 'AS_GIVEN':
                                        exp_data_transpose_current_orientation = data_transpose_return_orientation
                                        exp_data_transpose_return_orientation = data_transpose_return_orientation
                                    elif data_transpose_return_orientation == 'AS_GIVEN':
                                        exp_data_transpose_current_orientation = exp_data_transpose_given_orientation
                                        exp_data_transpose_return_orientation = exp_data_transpose_given_orientation

                                    if exp_data_transpose_return_orientation == exp_data_transpose_given_orientation:
                                        pass  # EXP_DATA_TRANSPOSE = EXP_DATA_TRANSPOSE FOR BOTH ARRAY AND SPARSE_DICT
                                    elif exp_data_transpose_return_orientation != exp_data_transpose_given_orientation:
                                        if exp_data_transpose_return_format == 'ARRAY':
                                            EXP_DATA_TRANSPOSE = EXP_DATA_TRANSPOSE.transpose()
                                        elif exp_data_transpose_return_format == 'SPARSE_DICT':
                                            EXP_DATA_TRANSPOSE = sd.sparse_transpose(EXP_DATA_TRANSPOSE)

                                # NOW HAVE EXP_DATA & EXP_DATA_TRANSPOSE IN CORRECT RETURN FORMAT & ORIENTATION

                            ############################################################################################################

                            # WIPE OUT DATA IF NOT RETURNED
                            if 'DATA' not in DUM_RETURN_OBJECTS:
                                EXP_DATA = None
                                exp_data_current_format = None
                                exp_data_return_format = None
                                exp_data_current_orientation = None
                                exp_data_return_orientation = None
                                # 12/13/22 DONT UNHASH THESE, WILL CAUSE ERRORS AGAINST act
                                # exp_data_given_format = None
                                # exp_data_given_orientation = None

                            # WIPE OUT DATA_TRANSPOSE IF NOT RETURNED
                            if 'DATA_TRANSPOSE' not in DUM_RETURN_OBJECTS:
                                EXP_DATA_TRANSPOSE = None
                                exp_data_transpose_current_format = None
                                exp_data_transpose_return_format = None
                                exp_data_transpose_current_orientation = None
                                exp_data_transpose_return_orientation = None
                                # 12/13/22 DONT UNHASH THESE, WILL CAUSE ERRORS AGAINST act
                                # exp_data_transpose_given_format = None
                                # exp_data_transpose_given_orientation = None


                            EXP_XTX = None
                            EXP_XTX_INV = None
                            EXP_TARGET = None
                            EXP_TARGET_TRANSPOSE = None
                            EXP_TARGET_AS_LIST = None

                            expected_output = (f'Expected output:\n',
                                  f'exp_this_module = MLObjectOrienter\n',
                                  f'exp_calling_module = {exp_calling_module}\n',
                                  f'exp_calling_fxn = {exp_calling_fxn}\n',

                                  f'exp_data_given_orientation = {exp_data_given_orientation}\n',
                                  f'exp_data_given_format = {exp_data_given_format}\n',
                                  f'exp_data_current_orientation = {exp_data_current_orientation}\n',
                                  f'exp_data_current_format = {exp_data_current_format}\n'
                                  f'exp_data_return_orientation = {exp_data_return_orientation}\n',
                                  f'exp_data_return_format = {exp_data_return_format}\n',
                                  f'exp_data_transpose_given_orientation = {exp_data_transpose_given_orientation}\n',
                                  f'exp_data_transpose_given_format = {exp_data_transpose_given_format}\n',
                                  f'exp_data_transpose_current_orientation = {exp_data_transpose_current_orientation}\n',
                                  f'exp_data_transpose_current_format = {exp_data_transpose_current_format}\n'
                                  f'exp_data_transpose_return_orientation = {exp_data_transpose_return_orientation}\n',
                                  f'exp_data_transpose_return_format = {exp_data_transpose_return_format}\n'

                                  #   f'EXP_DATA = \n{EXP_DATA}'
                                  # f'EXP_DATA_TRANSPOSE = \n{EXP_DATA_TRANSPOSE}'
                                  )

                            # print(*expected_output)

                            Dummy = mloo.MLObjectOrienter(
                                DATA=DATA,
                                # ENTER format & orientation for DATA AS IT WOULD BE IF DATA IS GIVEN, EVEN
                                # IF IS NOT GIVEN, IF A DATA OBJECT IS TO BE RETURNED
                                # DATA given_orientation MUST BE ENTERED IF ANY DATA OBJECT IS TO BE RETURNED
                                data_given_orientation=data_given_orientation,
                                data_return_orientation=data_return_orientation,
                                data_return_format=data_return_format,

                                DATA_TRANSPOSE=DATA_TRANSPOSE,
                                data_transpose_given_orientation=data_transpose_given_orientation,
                                data_transpose_return_orientation=data_transpose_return_orientation,
                                data_transpose_return_format=data_transpose_return_format,

                                XTX=None,
                                xtx_return_format=None,  # XTX IS SYMMETRIC SO DONT NEED ORIENTATION

                                XTX_INV=None,
                                xtx_inv_return_format=None,  # XTX IS SYMMETRIC SO DONT NEED ORIENTATION

                                ############################################################################################################
                                TARGET=None,
                                # ENTER format & orientation for TARGET AS IT WOULD BE IF TARGET IS GIVEN,
                                # EVEN IF IS NOT GIVEN, IF A TARGET OBJECT IS TO BE RETURNED
                                # TARGET given_orientation MUST BE ENTERED IF ANY TARGET OBJECT IS TO BE RETURNED
                                target_given_orientation=None,
                                target_return_orientation=None, #'AS_GIVEN',
                                target_return_format=None, #'AS_GIVEN',
                                target_is_multiclass=None, #False,
                                TARGET_TRANSPOSE=None,
                                target_transpose_given_orientation=None,
                                target_transpose_return_orientation=None,
                                target_transpose_return_format=None, #'AS_GIVEN',
                                TARGET_AS_LIST=None,
                                ############################################################################################################

                                RETURN_OBJECTS=DUM_RETURN_OBJECTS,
                                bypass_validation=bypass_validation,
                                calling_module=calling_module,
                                calling_fxn=calling_fxn)

                            ACT_DATA = Dummy.DATA
                            ACT_DATA_TRANSPOSE = Dummy.DATA_TRANSPOSE
                            ACT_XTX = Dummy.XTX
                            ACT_XTX_INV = Dummy.XTX_INV
                            ACT_TARGET = Dummy.TARGET
                            ACT_TARGET_TRANSPOSE = Dummy.TARGET_TRANSPOSE
                            ACT_TARGET_AS_LIST = Dummy.TARGET_AS_LIST

                            act_this_module = 'MLObjectOrienter'
                            act_calling_module = gmn.get_module_name(str(sys.modules[__name__]))
                            act_calling_fxn = 'guard_test'

                            act_data_given_orientation = Dummy.data_given_orientation
                            act_data_given_format = Dummy.data_given_format
                            act_data_current_orientation = Dummy.data_return_orientation
                            act_data_current_format = Dummy.data_current_format
                            act_data_return_orientation = Dummy.data_return_orientation
                            act_data_return_format = Dummy.data_return_format

                            act_data_transpose_given_orientation = Dummy.data_transpose_given_orientation
                            act_data_transpose_given_format = Dummy.data_transpose_given_format
                            act_data_transpose_current_orientation = Dummy.data_transpose_current_orientation
                            act_data_transpose_current_format = Dummy.data_transpose_current_format
                            act_data_transpose_return_orientation = Dummy.data_transpose_return_orientation
                            act_data_transpose_return_format = Dummy.data_transpose_return_format

                            NAMES =[
                                    'calling_module',
                                    'calling_fxn',
                                    'data_given_orientation',
                                    'data_given_format',
                                    'data_current_orientation',
                                    'data_current_format',
                                    'data_return_orientation',
                                    'data_return_format',
                                    'data_transpose_given_orientation',
                                    'data_transpose_given_format',
                                    'data_transpose_current_orientation',
                                    'data_transpose_current_format',
                                    'data_transpose_return_orientation',
                                    'data_transpose_return_format',
                                    'DATA',
                                    'DATA_TRANSPOSE',
                                    'XTX',
                                    'XTX_INV',
                                    'TARGET',
                                    'TARGET_TRANSPOSE',
                                    'TARGET_AS_LIST'
                                    ]

                            EXPECTED_INPUTS = [
                                    exp_calling_module,
                                    exp_calling_fxn,
                                    exp_data_given_orientation,
                                    exp_data_given_format,
                                    exp_data_current_orientation,
                                    exp_data_current_format,
                                    exp_data_return_orientation,
                                    exp_data_return_format,
                                    exp_data_transpose_given_orientation,
                                    exp_data_transpose_given_format,
                                    exp_data_transpose_current_orientation,
                                    exp_data_transpose_current_format,
                                    exp_data_transpose_return_orientation,
                                    exp_data_transpose_return_format,
                                    EXP_DATA,
                                    EXP_DATA_TRANSPOSE,
                                    EXP_XTX,
                                    EXP_XTX_INV,
                                    EXP_TARGET,
                                    EXP_TARGET_TRANSPOSE,
                                    EXP_TARGET_AS_LIST
                                    ]



                            ACTUAL_OUTPUTS = [
                                    act_calling_module,
                                    act_calling_fxn,
                                    act_data_given_orientation,
                                    act_data_given_format,
                                    act_data_current_orientation,
                                    act_data_current_format,
                                    act_data_return_orientation,
                                    act_data_return_format,
                                    act_data_transpose_given_orientation,
                                    act_data_transpose_given_format,
                                    act_data_transpose_current_orientation,
                                    act_data_transpose_current_format,
                                    act_data_transpose_return_orientation,
                                    act_data_transpose_return_format,
                                    ACT_DATA,
                                    ACT_DATA_TRANSPOSE,
                                    ACT_XTX,
                                    ACT_XTX_INV,
                                    ACT_TARGET,
                                    ACT_TARGET_TRANSPOSE,
                                    ACT_TARGET_AS_LIST
                                    ]

                            for description, expected_thing, actual_thing in zip(NAMES, EXPECTED_INPUTS, ACTUAL_OUTPUTS):

                                try:
                                    is_equal = np.array_equiv(expected_thing, actual_thing)
                                    # print(f'\033[91m\n*** TEST EXCEPTED ON np.array_equiv METHOD ***\033[0m\x1B[0m\n')
                                except:
                                    try:
                                        is_equal = expected_thing == actual_thing
                                    except:
                                        print(f'{description}:')
                                        print(f'\n\033[91mEXP_OBJECT = \n{expected_thing}\033[0m\x1B[0m\n')
                                        print(f'\n\033[91mACT_OBJECT = \n{actual_thing}\033[0m\x1B[0m\n')
                                        raise Exception(f'\n*** TEST FAILED "==" METHOD ***\n')

                                if not is_equal:
                                    print(f'\n\033[91mEXP_OBJECT = \n{expected_thing}\033[0m\x1B[0m\n')
                                    print(f'\n\033[91mACT_OBJECT = \n{actual_thing}\033[0m\x1B[0m\n')
                                    print(f'\nFailed on trial {ctr} of at most {total_trials}')

                                    print(f'DATA is {"not given" if DATA is None else "given"}, '
                                          f'DATA_TRANSPOSE is {"not given" if DATA_TRANSPOSE is None else "given"}.')
                                    print(*expected_output)

                                    raise Exception(
                                        f'\n*** {description} FAILED EQUALITY TEST, \nexpected = \n{expected_thing}\n'
                                        f'actual = \n{actual_thing} ***\n')
                                else:
                                    pass  # print(f'\033[92m     *** {description} PASSED ***\033[0m\x1B[0m')


    print(f'\n\033[92m*** TEST DONE. ALL PASSED. ***\033[0m\x1B[0m\n')





