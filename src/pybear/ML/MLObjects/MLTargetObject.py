import numpy as np
import sys, inspect
import sparse_dict as sd
from data_validation import arg_kwarg_validater as akv
from debug import get_module_name as gmn
from MLObjects import MLObject as mlo


# 12/16/22 THIS MODULE HANDLES "TARGET" AS OPPOSED TO "TARGET_TRANSPOSE".  THE TWO CLASSES EXIST TO MANAGE THE VICE-VERSA
# TYPE RELATIONSHIP OF VALIDATION FOR IS/ISNT MULTI-CLASS


class MLTargetObject(mlo.MLObject):
    '''Child class for handling TARGET within MLReqression operations. May be multi-class. Object is accessed as attribute of class.'''

    def __init__(self, OBJECT, given_orientation, return_orientation='AS_GIVEN', return_format='AS_GIVEN',
                 is_multiclass=False, bypass_validation=False, calling_module=None, calling_fxn=None):

        self.is_multiclass = is_multiclass   # WILL BE OVERWROTE by object_init IF not bypass_validation

        super().__init__(OBJECT, given_orientation, name='TARGET', return_orientation=return_orientation, return_format=return_format,
                         bypass_validation=bypass_validation, calling_module=calling_module, calling_fxn=calling_fxn)

        # OVERWRITE FROM PARENT init
        self.this_module = gmn.get_module_name(str(sys.modules[__name__]))

    # INHERITS
    # _exception()        Exception handling template.
    # object_init()       Unique OBJECT init sequence.
    # get_shape()         Return OBJECT shape like numpy.
    # _transpose()        Transpose OBJECT attribute.
    # get_transpose()     Return transpose of OBJECT in its current state.
    # to_array()          Convert OBJECT attribute to a array.
    # to_dict()           Convert OBJECT attribute to a sparse dict.
    # return_as_array()   Return OBJECT in current state as array.
    # return_as_dict()    Return OBJECT in current state as sparse dict.
    # return_as_column()  Return OBJECT in current state oriented as column.
    # return_as_row()     Return OBJECT in current state oriented as row.
    # is_equiv()          Return boolean np.array_equiv or sd.sparse_equiv of this class's OBJECT with another MLObject class's OBJECT.

    # OVERWRITES
    def object_init(self):
        """This happens in init if not bypass_validation."""

        fxn = inspect.stack()[0][3]
        self.is_multiclass = akv.arg_kwarg_validater(self.is_multiclass, 'is_multiclass', [True, False], self.this_module, fxn)

        is_error, error_text = False, ''
        if self.is_multiclass:
            error_text = f'Given TARGET OBJECT appears to be single class but is declared as multi-class'
            if self.given_orientation == 'COLUMN' and self.outer_len == 1: is_error = True
            elif self.given_orientation == 'ROW' and self.inner_len == 1: is_error = True
        elif not self.is_multiclass:
            error_text = f'Given TARGET OBJECT appears to be multi-class but is declared as single class'
            if self.given_orientation == 'COLUMN' and self.outer_len > 1: is_error = True
            elif self.given_orientation == 'ROW' and self.inner_len > 1: is_error = True
        else: self._exception(fxn, f'is_multiclass is not boolean')

        if is_error: self._exception(fxn, error_text)
        del is_error, error_text

    # OVERWRITES
    def return_XTX(self):
        """Return XTX calculated from OBJECT."""
        pass

    # OVERWRITES
    def return_XTX_INV(self):
        """Return XTX_INV calculated from OBJECT."""
        pass





if __name__ == '__main__':

    # TEST MODULE --- TEST CODE VERIFIED GOOD 12/16/22
    exp_this_module = gmn.get_module_name(str(sys.modules[__name__]))
    exp_calling_module = 'MLTargetObject'
    exp_calling_fxn = 'guard_test'

    expected_trials = 4*4*2*2*2*2
    ctr = 0
    for (raw_columns, raw_rows) in ((100, 2), (2, 100), (40, 60), (60, 40)):   # LONG & THIN TO ACTUATE CONDITIONAL zip/unzip

        OBJECT1 = np.arange(1, 1+raw_columns*raw_rows).reshape((raw_columns, raw_rows))
        OBJECT2 = sd.zip_list_as_py_float(OBJECT1)
        OBJECT3 = np.random.randint(0, 10, (raw_columns, raw_rows))
        OBJECT4 = sd.zip_list_as_py_float(OBJECT3)

        for exp_is_multiclass, exp_given_format, GIVEN_OBJECT in \
                            zip(
                            (False, False, True, True),
                            ('ARRAY', 'SPARSE_DICT', 'ARRAY', 'SPARSE_DICT'),
                            (OBJECT1, OBJECT2, OBJECT3, OBJECT4)
                            ):
            for exp_given_orientation in ['COLUMN', 'ROW']:
                exp_columns = raw_columns if exp_given_orientation == 'COLUMN' else raw_rows
                exp_rows = raw_columns if exp_given_orientation == 'ROW' else raw_rows
                for exp_return_orientation in ['COLUMN', 'ROW']:

                    exp_current_orientation = exp_return_orientation
                    exp_shape = (raw_columns, raw_rows)
                    exp_outer_len = raw_columns
                    exp_inner_len = raw_rows
                    if exp_return_orientation != exp_given_orientation:
                        exp_shape = (raw_rows, raw_columns)
                        exp_outer_len, exp_inner_len = exp_inner_len, exp_outer_len

                    if exp_return_orientation == 'COLUMN' and exp_outer_len > 1: exp_is_multiclass = True
                    elif exp_return_orientation == 'ROW' and exp_inner_len > 1: exp_is_multiclass = True

                    for exp_return_format in ['ARRAY', 'SPARSE_DICT']:
                        exp_current_format = exp_return_format
                        for exp_bypass_validation in [True, False]:
                            ctr += 1
                            print(f'\n' + f'*'*70 + f'\nRunning trial {ctr} of {expected_trials}...')

                            if exp_return_format == 'ARRAY':
                                if exp_given_format == 'ARRAY': EXP_OBJECT = GIVEN_OBJECT
                                elif exp_given_format == 'SPARSE_DICT': EXP_OBJECT = sd.unzip_to_ndarray_float64(GIVEN_OBJECT)[0]
                                if exp_return_orientation != exp_given_orientation: EXP_OBJECT = EXP_OBJECT.transpose()
                            elif exp_return_format == 'SPARSE_DICT':
                                if exp_given_format == 'ARRAY': EXP_OBJECT = sd.zip_list_as_py_float(GIVEN_OBJECT)
                                elif exp_given_format == 'SPARSE_DICT': EXP_OBJECT = GIVEN_OBJECT
                                if exp_return_orientation != exp_given_orientation: EXP_OBJECT = sd.sparse_transpose(EXP_OBJECT)

                            DummyObject = MLTargetObject(GIVEN_OBJECT,
                                                         exp_given_orientation,
                                                         exp_return_orientation,
                                                         exp_return_format,
                                                         is_multiclass=exp_is_multiclass,
                                                         bypass_validation=exp_bypass_validation,
                                                         calling_module=exp_calling_module,
                                                         calling_fxn=exp_calling_fxn)

                            ACT_OBJECT = DummyObject.OBJECT
                            act_given_format = DummyObject.given_format
                            act_given_orientation = DummyObject.given_orientation
                            act_current_orientation = DummyObject.current_orientation
                            act_current_format = DummyObject.current_format
                            act_return_format = DummyObject.return_format
                            act_return_orientation = DummyObject.return_orientation
                            act_is_multiclass = DummyObject.is_multiclass
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
                                    'return_orientation',
                                    'is_multiclass',
                                    'return_format',
                                    'bypass_validation',
                                    'given_orientation',
                                    'given_format',
                                    'current_orientation',
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
                                    exp_return_orientation,
                                    exp_is_multiclass,
                                    exp_return_format,
                                    exp_bypass_validation,
                                    exp_given_orientation,
                                    exp_given_format,
                                    exp_current_orientation,
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
                                    act_return_orientation,
                                    act_is_multiclass,
                                    act_return_format,
                                    act_bypass_validation,
                                    act_given_orientation,
                                    act_given_format,
                                    act_current_orientation,
                                    act_current_format
                            ]

                            print(f'Expected output:\n'
                                    f'exp_this_module = {exp_this_module}\n',
                                    f'exp_calling_module = {exp_calling_module}\n',
                                    f'exp_calling_fxn = {exp_calling_fxn}\n',
                                    f'exp_columns = {exp_columns}\n',
                                    f'exp_rows = {exp_rows}\n',
                                    f'exp_shape = {exp_shape}\n',
                                    f'exp_outer_len = {exp_outer_len}\n',
                                    f'exp_inner_len = {exp_inner_len}\n',
                                    f'exp_return_orientation = {exp_return_orientation}\n',
                                    f'exp_is_multiclass = {exp_is_multiclass}\n',
                                    f'exp_return_format = {exp_return_format}\n',
                                    f'exp_given_orientation = {exp_given_orientation}\n',
                                    f'exp_given_format = {exp_given_format}\n',
                                    f'exp_current_orientation = {exp_current_orientation}\n',
                                    f'exp_current_format = {exp_current_format}\n')

                            for description, expected_thing, actual_thing in zip(DESCRIPTIONS, EXP_OBJS, ACT_OBJS):

                                try:
                                    is_equal = np.array_equiv(expected_thing, actual_thing)
                                    # print(f'\033[91m\n*** TEST EXCEPTED ON np.array_equiv METHOD ***\033[0m\x1B[0m\n')
                                except:
                                    try: is_equal = expected_thing == actual_thing
                                    except:
                                        print(f'\033[91mEXP_OBJECT = \n{EXP_OBJECT}\033[0m\x1B[0m\n')
                                        print(f'\n\033[91mACT_OBJECT = \n{ACT_OBJECT}\033[0m\x1B[0m\n')
                                        raise Exception(f'\n\033[91*** TEST FAILED "==" METHOD ***\033[0m\x1B[0m\n')

                                if not is_equal:
                                    print(f'\033[91mEXP_OBJECT = \n{EXP_OBJECT}\033[0m\x1B[0m\n')
                                    print(f'\n\033[91mACT_OBJECT = \n{ACT_OBJECT}\033[0m\x1B[0m\n')
                                    raise Exception(f'\n*** {description} FAILED EQUALITY TEST, expected = {expected_thing}, '
                                                    f'actual = {actual_thing} ***\n')
                                else: pass # print(f'\033[92m     *** {description} PASSED ***\033[0m\x1B[0m')


    print(f'\n\033[92m*** TEST DONE. ALL PASSED. ***\033[0m\x1B[0m\n')


















