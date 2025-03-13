import numpy as np
import sys
import sparse_dict as sd
from debug import get_module_name as gmn
from MLObjects import MLObject as mlo


# METHODS
# __init__()          Parent class for handling DATA and XTX within MLReqression operations. Object is accessed as attribute of class.
# _exception()        Exception handling template.
# object_init()       Unique OBJECT init sequence.
# get_shape()         Return OBJECT shape like numpy.
# to_array()          Convert OBJECT attribute to a array.
# to_dict()           Convert OBJECT attribute to a sparse dict.
# return_as_array()   Return OBJECT in current state as array.
# return_as_dict()    Return OBJECT in current state as sparse dict.
# is_equiv()          Return boolean np.array_equiv or sd.sparse_equiv of this class's OBJECT with another MLObject class's OBJECT.
# invert()            Invert OBJECT as attribute of this class.
# return_inverse()    Returns inverse of OBJECT as array without changing the class attribute.


# ATTRIBUTES
# _columns
# _rows
# bypass_validation
# calling_module
# calling_fxn
# current_format
# given_format
# inner_len
# OBJECT
# outer_len
# return_format
# this_module


class MLObjectSymmetric(mlo.MLObject):
    '''Child class for handling XTX within MLReqression operations. Object is accessed as attribute of class.'''
    def __init__(self, OBJECT, return_format='AS_GIVEN', bypass_validation=False, calling_module=None, calling_fxn=None):

        # DUMMIES TO FAKE OUT PARENT > given_orientation='ROW', return_orientation='ROW'

        super().__init__(OBJECT, 'ROW', name='XTX', return_orientation='ROW', return_format=return_format,
                         bypass_validation=bypass_validation, calling_module=calling_module, calling_fxn=calling_fxn)

        # OVERWRITE FROM PARENT init
        self.this_module = gmn.get_module_name(str(sys.modules[__name__]))

    # INHERITS
    # _exception()
    # object_init()

    # OVERWRITES
    def get_shape(self):
        '''Return OBJECT shape like numpy.'''
        if self.current_format == 'ARRAY': self.outer_len = len(self.OBJECT)
        elif self.current_format == 'SPARSE_DICT': self.outer_len = sd.outer_len(self.OBJECT)

        self.inner_len = self.outer_len

        return self.outer_len, self.inner_len

    # OVERWRITES
    def _transpose(self):
        '''Transpose OBJECT attribute.'''
        pass

    # OVERWRITES
    def get_transpose(self):
        """Return transpose of OBJECT in its current state without changing state."""
        pass

    # INHERITS
    # to_array()
    # to_dict()
    # return_as_array()
    # return_as_dict()

    # OVERWRITES
    def return_as_column(self):
        pass

    # OVERWRITES
    def return_as_row(self):
        pass

    # OVERWRITES
    def return_XTX(self):
        """Return XTX calculated from OBJECT."""
        pass

    # OVERWRITES
    def return_XTX_INV(self):
        """Return XTX_INV calculated from OBJECT."""
        pass

    # INHERITS
    # is_equiv()

    # UNIQUE
    def invert(self):
        '''Invert OBJECT as attribute of this class.'''
        if self.current_format == 'ARRAY':
            self.OBJECT = np.linalg.inv(self.OBJECT)
        elif self.current_format == 'SPARSE_DICT':
            self.OBJECT = sd.zip_list_as_py_float(np.linalg.inv(sd.unzip_to_ndarray_float64(self.OBJECT)[0]))


    def return_inverse(self):
        """Returns inverse of OBJECT as array without changing the class attribute."""
        if self.current_format == 'ARRAY': return np.linalg.inv(self.OBJECT)
        elif self.current_format == 'SPARSE_DICT':
            return sd.zip_list_as_py_float(np.linalg.inv(sd.unzip_to_ndarray_float64(self.OBJECT)[0]))








if __name__ == '__main__':

    # TEST MODULE --- TEST CODE VERIFIED GOOD 12/8/22
    exp_this_module = gmn.get_module_name(str(sys.modules[__name__]))
    exp_calling_module = 'MLObjectSymmetric'
    exp_calling_fxn = 'guard_test'

    dum_cols = 500
    dum_rows = 5000

    DUM_OBJECT = np.random.randint(0,10,(dum_rows,dum_cols))

    exp_columns = dum_cols
    exp_rows = dum_cols
    exp_shape = (dum_cols, dum_cols)
    exp_outer_len = dum_cols
    exp_inner_len = dum_cols

    print(f'Building np XTX Object...')
    OBJECT1 = np.matmul(DUM_OBJECT.transpose(), DUM_OBJECT)
    print(f'Done.\n')
    print(f'Building sd XTX Object...')
    OBJECT2 = sd.zip_list_as_py_float(OBJECT1)
    print(f'Done.\n')

    expected_trials = 2*2*2
    ctr = 0
    for exp_given_format, GIVEN_OBJECT in zip(('ARRAY', 'SPARSE_DICT'), (OBJECT1, OBJECT2)):

        for exp_return_format in ['ARRAY', 'SPARSE_DICT']:
            exp_current_format = exp_return_format
            for exp_bypass_validation in [True, False]:
                ctr += 1
                print(f'\n' + f'*'*70 + f'\nRunning trial {ctr} of {expected_trials}...')

                if exp_return_format == 'ARRAY':
                    if exp_given_format == 'ARRAY': EXP_OBJECT = GIVEN_OBJECT
                    elif exp_given_format == 'SPARSE_DICT': EXP_OBJECT = sd.unzip_to_ndarray_float64(GIVEN_OBJECT)[0]
                elif exp_return_format == 'SPARSE_DICT':
                    if exp_given_format == 'ARRAY': EXP_OBJECT = sd.zip_list_as_py_float(GIVEN_OBJECT)
                    elif exp_given_format == 'SPARSE_DICT': EXP_OBJECT = GIVEN_OBJECT

                DummyObject = MLObjectSymmetric(GIVEN_OBJECT,
                                                 exp_return_format,
                                                 bypass_validation=exp_bypass_validation,
                                                 calling_module=exp_calling_module,
                                                 calling_fxn=exp_calling_fxn)

                ACT_OBJECT = DummyObject.OBJECT
                act_given_format = DummyObject.given_format
                act_current_format = DummyObject.current_format
                act_return_format = DummyObject.return_format
                act_bypass_validation = DummyObject.bypass_validation
                act_calling_module = DummyObject.calling_module
                act_calling_fxn = DummyObject.calling_fxn
                act_this_module = DummyObject.this_module
                act_columns = DummyObject._columns
                act_rows = DummyObject._rows
                act_shape = DummyObject.get_shape()
                act_outer_len = DummyObject.outer_len
                act_inner_len = DummyObject.inner_len

                DESCRIPTIONS = \
                    [
                        'this_module',
                        'calling_module',
                        'calling_fxn',
                        'OBJECT',
                        'columns',
                        'rows',
                        'shape',
                        'outer_len',
                        'inner_len',
                        'return_format',
                        'bypass_validation',
                        'given_format',
                        'current_format'
                ]

                EXP_OBJS = \
                    [
                        exp_this_module,
                        exp_calling_module,
                        exp_calling_fxn,
                        EXP_OBJECT,
                        exp_columns,
                        exp_rows,
                        exp_shape,
                        exp_outer_len,
                        exp_inner_len,
                        exp_return_format,
                        exp_bypass_validation,
                        exp_given_format,
                        exp_current_format
                ]

                ACT_OBJS = \
                    [
                        act_this_module,
                        act_calling_module,
                        act_calling_fxn,
                        ACT_OBJECT,
                        act_columns,
                        act_rows,
                        act_shape,
                        act_outer_len,
                        act_inner_len,
                        act_return_format,
                        act_bypass_validation,
                        act_given_format,
                        act_current_format
                ]

                print(f'Expected output:\n'
                        f'exp_this_module = {exp_this_module}\n',
                        f'exp_calling_module = {exp_calling_module}\n',
                        f'exp_calling_fxn = {exp_calling_fxn}\n',
                        # f'EXP_OBJECT = {EXP_OBJECT}\n',
                        f'exp_columns = {exp_columns}\n',
                        f'exp_rows = {exp_rows}\n',
                        f'exp_shape = {exp_shape}\n',
                        f'exp_outer_len = {exp_outer_len}\n',
                        f'exp_inner_len = {exp_inner_len}\n',
                        f'exp_return_format = {exp_return_format}\n',
                        f'exp_given_format = {exp_given_format}\n',
                        f'exp_current_format = {exp_current_format}\n')


                for description, expected_thing, actual_thing in zip(DESCRIPTIONS, EXP_OBJS, ACT_OBJS):

                    try:
                        is_equal = np.array_equiv(expected_thing, actual_thing)
                        # print(f'\033[91m\n*** TEST EXCEPTED ON np.array_equiv METHOD ***\033[0m\x1B[0m\n')
                    except:
                        try: is_equal = expected_thing == actual_thing
                        except: raise Exception(f'\n*** TEST FAILED "==" METHOD ***\n')

                    if not is_equal:
                        raise Exception(f'\n*** {description} FAILED EQUALITY TEST, expected = {expected_thing}, '
                                        f'actual = {actual_thing} ***\n')
                    else: print(f'\033[92m     *** {description} PASSED ***\033[0m\x1B[0m')


    print(f'\n\033[92m*** TEST DONE. ALL PASSED. ***\033[0m\x1B[0m\n')















