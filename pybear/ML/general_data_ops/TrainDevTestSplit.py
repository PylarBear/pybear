import sys, inspect, time
from general_sound import winlinsound as wls
import numpy as np
from debug import get_module_name as gmn
from general_list_ops import list_select as ls
from general_data_ops import get_shape as gs, train_dev_test_split_core as tdtsc, new_np_random_choice as nnrc
from data_validation import arg_kwarg_validater as akv, validate_user_input as vui
from ML_PACKAGE._data_validation import list_dict_validater as ldv
from MLObjects.SupportObjects import master_support_object_dict as msod, validate_full_support_object as vfso
from MLObjects import MLRowColumnOperations as mlrco, MLObject as mlo
from MLObjects.PrintMLObject import SmallObjectPreview as sop


# self.TRAIN, self.DEV, self.TEST ARE RETURNED, BUT self.DATA, self.TARGET, self.REFVECS CAN BE ACCESSED AS ATTRS

# self.DATA, self.TARGET, self.REFVECS LOOK LIKE:
# 2 WAY SPLIT, EG FOR DATA:  (DATA_TRAIN, DATA_TEST)
# 3 WAY SPLIT, EG FOR TARGET:  (TARGET_TRAIN, TARGET_DEV, TARGET_TEST)

# self.TRAIN, self.DEV, self.TEST LOOK LIKE:
# FOR 2 OBJECTS (DATA & TARGET IN THIS CASE), EG self.TRAIN:  (DATA_TRAIN, TARGET_TRAIN)
# FOR 3 OBJECTS, EG self.DEV:  (DATA_DEV, TARGET_DEV, REFVECS_DEV)

# self.TRAIN, OPTIONALLY self.DEV, AND self.TEST ARE return
# 2 WAY SPLIT OF 2 OBJECTS RETURNS, EG:  (DATA_TRAIN, TARGET_TRAIN), (DATA_TEST, TARGET_TEST)
# 3 WAY SPLIT OF 3 OBJECTS RETURNS, EG:  (DATA_TRAIN, TARGET_TRAIN, REFVECS_TRAIN), (DATA_DEV, TARGET_DEV, REFVECS_DEV), (DATA_TEST, TARGET_TEST, REFVECS_TEST)

# EACH OF mask(), random(), partition(), or category() GENERATE MASKS (SILENTLY IF PASSED THE INFORMATION NEEDED OR VERBOSELY VIA
# PROMPTS FOR INFORMATION IF IT IS NOT GIVEN) WHICH ARE THEN APPLIED OVER THE FULL DATA SET VIA core_run_fxn (WHICH CALLS
# train_dev_test_split_core()) TO return TRAIN & TEST SETS OR TRAIN & DEV & TEST SETS



# Parent of MLPackageTrainDevTestSplitOLDSUPOBJS, MLPackageTrainDevTestSplitNEWSUPOBJS
# THE ONLY 2 PLACES WHERE SUPOBJS ARE BEING USED IS FOR category METHOD (SUPOBJS DONT CHANGE FOR A TRAIN-DEV-TEST SPLIT),
#       1) TO CREATE A PRETTY PRINT OUT FOR USER TO PICK OBJ/COL/VALUE TO MOVE OUT OF TRAIN & INTO DEV/TEST
#       2) TO PICK DEV OR TEST FILTERING CATEGORY BY COLUMN NAME
class TrainDevTestSplit:
    '''Dont call this directly, call with mask(), random(), partition(), or cateyory() methods only.'''
    def __init__(self, DATA=None, TARGET=None, REFVECS=None, data_given_orientation=None, target_given_orientation=None,
                 refvecs_given_orientation=None, bypass_validation=None):

        self._exception = lambda words, fxn=None: \
            Exception(f'{self.this_module}{f".{fxn}()" if not fxn is None else f""} >>> {words}')

        self.this_module = gmn.get_module_name(str(sys.modules[__name__]))
        fxn = '__init__'

        self.bypass_validation = akv.arg_kwarg_validater(bypass_validation, 'bypass_validation', [True,False,None],
                                                         self.this_module, fxn, return_if_none=False)

        # IRREGARDLESS OF bypass_validation, VERIFY ARE LEGIT OBJECTS & GET FORMAT
        self.data_given_format, self.DATA = ldv.list_dict_validater(DATA, 'DATA')
        self.target_given_format, self.TARGET = ldv.list_dict_validater(TARGET, 'TARGET')
        self.refvec_given_format, self.REFVECS = ldv.list_dict_validater(REFVECS, 'REFVECS')

        self.data_given_orientation = akv.arg_kwarg_validater(data_given_orientation,
                                        'data_given_orientation', ['ROW','COLUMN',None], self.this_module, fxn)
        self.target_given_orientation = akv.arg_kwarg_validater(target_given_orientation,
                                        'target_given_orientation', ['ROW', 'COLUMN', None], self.this_module, fxn)
        self.refvecs_given_orientation = akv.arg_kwarg_validater(refvecs_given_orientation,
                                        'refvecs_given_orientation', ['ROW','COLUMN',None], self.this_module, fxn)

        # BUILD HELPER TUPLES FOR FAST VALIDATION ######################################################################
        NAMES = ('DATA', 'TARGET', 'REFVECS')
        OBJECTS = (self.DATA, self.TARGET, self.REFVECS)
        ORIENTATIONS = (self.data_given_orientation, self.target_given_orientation, self.refvecs_given_orientation)

        if not self.bypass_validation:
            if False not in map(lambda x: x is None, OBJECTS):    # IF ALL OBJECTS ARE None
                raise self._exception(f'AT LEAST ONE OBJECT MUST BE PASSED AS KWARG', fxn)

            for name, _obj, _orient in zip(NAMES, OBJECTS, ORIENTATIONS):
                if not _obj is None and _orient is None:
                    raise self._exception(f'PASSED {name} OBJECT MUST ALSO HAVE A given_orientation KWARG', fxn)

        # USE HELPER OBJECT TO DETERMINE ACTIVE (aka PASSED) OBJECTS --- CAN BE 1,2,3 OBJECTS
        _range = range(len(OBJECTS))
        ACTVS = lambda OBJ: tuple(OBJ[idx] for idx in _range if not OBJECTS[idx] is None)
        self.ACTV_NAMES = ACTVS(NAMES)
        self.ACTV_OBJECTS = ACTVS(OBJECTS)
        self.ACTV_ORIENTATIONS = ACTVS(ORIENTATIONS)
        del NAMES, OBJECTS, ORIENTATIONS, _range, ACTVS

        # VERIFY ACTIVE OBJECTS GAVE EQUAL NUMBER OF ROWS
        ROWS = list(gs.get_shape(_a, _b, _c)[0] for _a,_b,_c in zip(self.ACTV_NAMES, self.ACTV_OBJECTS, self.ACTV_ORIENTATIONS))
        if not self.bypass_validation and not min(ROWS) == max(ROWS):
            raise self._exception(f'PASSED OBJECTS DO NOT HAVE EQUAL NUMBER OF ROWS ({", ".join(map(str, ROWS))})', fxn)

        self.rows = ROWS[0]
        del ROWS

        # HELPERS FOR category()
        self.data_cols = gs.get_shape('DATA', DATA, self.data_given_orientation)[1] if 'DATA' in self.ACTV_NAMES else None
        self.target_cols = gs.get_shape('TARGET', TARGET, self.target_given_orientation)[1] if 'TARGET' in self.ACTV_NAMES else None
        self.refvec_cols = gs.get_shape('REFVECS', REFVECS, self.refvecs_given_orientation)[1] if 'REFVECS' in self.ACTV_NAMES else None

        # PLACEHOLDERS
        self.MASK = None

    # END init #########################################################################################################
    ####################################################################################################################
    ####################################################################################################################


    def mask(self, MASK):
        # SILENT METHOD ONLY
        # MASK MUST BE GENERATED EXTERNALLY AND PASSED AS LIST-TYPE OF 1 (FOR RETURN 2) OR 2 COLUMNS (FOR RETURN 3)
        # ORIENTED AS [[]=COLUMN].
        # MASK ENTRIES MUST BE BIN OR BOOL, True INDICATING KEEP THAT ROW FOR THAT OBJECT.  MAX ONE TRUE PER ROW.
        self.MASK = MASK
        return self.core_run_fxn()    # LET train_dev_test_split_core HANDLE VALIDATION


    def random(self, dev_count=None, dev_percent=None, test_count=None, test_percent=None):
        '''Generate Train-Test or Train-Dev-Test split by random draw.'''
        # SILENT (ONE OR TWO KWARG PASSED) OR VERBOSE (NO KWARGS PASSED)
        # IF ONE DEV PASSED: TWO-WAY PULL, IF ONE TEST PASSED: A TWO-WAY PULL, IF ONE DEV & ONE TEST PASSED: 3-WAY PULL
        # THE GOAL OF THIS MODULE IS TO GET THE INFORMATION TO BUILD A 1-COLUMN OR 2-COLUMN MASK, THEN THE MASK IS PASSED
        # TO core_run_fxn()

        fxn = inspect.stack()[0][3]

        def approver(func, ARGS):
            while True:
                output = func(*ARGS)
                if vui.validate_user_str(f'User entered {output}, Accept? (y/n) > ', 'YN') == 'Y': return output

        def count_validator(_count, name, min_allowed, max_allowed):
            if not _count in range(min_allowed, max_allowed):
                raise self._exception(f'{name}_count MUST BE AN INTEGER >= {min_allowed} AND <= {max_allowed}')

        def percent_validator(_percent, name, min_allowed, max_allowed):
            # PASS ABSOLUTE ROWS TO min_ & max_allowed, NOT A %
            _, __ = 100 * min_allowed / self.rows, 100 * max_allowed / self.rows
            if not _percent >= _ and not _percent <= __:
                raise self._exception(f'{name}_percent MUST BE >= {_: .2f} AND <= {__, :.2f}', fxn)
            del _, __

        # A ZERO IS PASSED, CHANGE IT TO None
        zero_setter = lambda kwarg: None if kwarg==0 else kwarg
        dev_count, dev_percent, test_count, test_percent = tuple(map(zero_setter, (dev_count, dev_percent, test_count, test_percent)))
        del zero_setter

        if not self.bypass_validation:
            # CANT ENTER count AND percent FOR dev AND test
            if not dev_count is None and not dev_percent is None:
                raise self._exception(f'If silently generating dev split, enter count or percent, not both', fxn=fxn)

            if not test_count is None and not test_percent is None:
                raise self._exception(f'If silently generating test split, enter count or percent, not both', fxn=fxn)

        #####################################################################################################################################
        # IF ALL random() KWARGS ARE NONE: VERBOSE SPLIT SETUP ##############################################################################
        if False not in map(lambda x: x is None, (dev_count, dev_percent, test_count, test_percent)):
            row_text = f'({self.rows} rows)'
            get_method = lambda name: vui.validate_user_str(f'\nSelect percent(p) or count(c) to determine {name} split size {row_text} > ', 'CP')
            get_percent = lambda name: vui.validate_user_float(f'\nEnter percent to randomly pull for {name} {row_text} > ', 100*1/self.rows, 100 * (self.rows-2) / self.rows)
            get_count = lambda name: vui.validate_user_int(f'\nEnter count to randomly pull for {name} {row_text} > ', min=1, max=self.rows-2)

            while True:
                dev_rows, test_rows = None, None
                pull_setup = vui.validate_user_str(f'Split TEST only(t) or split DEV and TEST(b) > ', 'TB')

                if pull_setup == 'T': dev_method, test_method  = None, approver(get_method, ('TEST',))
                elif pull_setup == 'B': dev_method, test_method = approver(get_method, ('DEV',)), approver(get_method, ('TEST',))
                del pull_setup

                # ONLY DO DEV IF 3 WAY SPLIT
                if not dev_method is None:
                    if dev_method == 'P': dev_rows = int(np.floor(self.rows * approver(get_percent, ('DEV',)) / 100))
                    elif dev_method == 'C': dev_rows = int(approver(get_count, ('DEV',)))
                # else: dev_rows STAYS None

                # MUST DO TEST METHOD WHETHER 2 OR 3 WAY SPLIT
                if test_method == 'P': test_rows = int(np.floor(self.rows * approver(get_percent, ('TEST',)) / 100))
                elif test_method == 'C': test_rows = int(approver(get_count, ('TEST',)))

                train_rows = self.rows - (0 if dev_rows is None else dev_rows) - test_rows

                if train_rows < 1:
                    print(f'\n*** THE DEV/TEST SIZES ENTERED ARE >= THE NUMBER OF EXAMPLES. TRY AGAIN. ***')
                    continue

                print(f'\nUser chose {train_rows} train examples ({100*train_rows/self.rows:.1f}%), '
                       + [f'{dev_rows} dev examples ({100*dev_rows/self.rows:.1f}%), ' if not dev_rows is None else 'no dev examples, '][0]
                       + f' and {test_rows} test examples ({100*test_rows/self.rows:.1f}%)')

                if vui.validate_user_str(f'Accept? (y/n) > ', 'YN') == 'Y':
                    del get_method, get_percent, get_count
                    break

            if dev_rows is None:  # test_rows MUST NOT BE None
                # DRAW RAND IDXS BASED ON test_rows
                TEST_IDXS = nnrc.new_np_random_choice(range(self.rows), (1, test_rows), replace=False).reshape((1, -1))[0]
                # BUILD TEST_MASK FIRST
                self.MASK = np.zeros(self.rows, dtype=bool)
                self.MASK[TEST_IDXS] = True
                del dev_rows, test_rows, TEST_IDXS

                self.MASK = self.MASK.reshape((1,-1)).astype(bool)

            elif not dev_rows is None:  # test_rows MUST NOT BE None
                # DRAW RAND IDXS FOR TOTAL OF dev_rows & test_rows, THEN FROM THAT RANDOMLY DRAW THE TEST SET (AND WHAT IS LEFT
                # IS THE DEV SET)  CANT PULL SEPARATELY, CANT ALLOW THE CHANCE THAT SAME ROW IS PULLED TWICE
                DEV_AND_TEST_IDXS = nnrc.new_np_random_choice(range(self.rows), (1, dev_rows + test_rows), replace=False).reshape((1, -1))[0]
                TEST_IDXS = nnrc.new_np_random_choice(DEV_AND_TEST_IDXS, (1, test_rows), replace=False).reshape((1, -1))[0]
                # BUILD TEST_MASK FIRST (IS FINAL, AND NEEDED TO MAKE DEV_MASK)
                TEST_MASK = np.zeros(self.rows, dtype=bool)
                TEST_MASK[TEST_IDXS] = True
                del dev_rows, test_rows, TEST_IDXS
                # BUILD DEV FROM DEV_AND_TEST_IDXS AND TEST_MASK
                DEV_MASK = np.zeros(self.rows, dtype=bool)
                DEV_MASK[DEV_AND_TEST_IDXS] = True
                DEV_MASK[TEST_MASK] = False
                del DEV_AND_TEST_IDXS

                # NOW JUST vstack DEV_MASK AND TEST_MASK TO GET A 2-COLUMN self.MASK
                self.MASK = np.vstack((DEV_MASK, TEST_MASK)).astype(bool)

        # END IF ALL random () KWARGS ARE NONE: VERBOSE SPLIT SETUP #########################################################################
        #####################################################################################################################################

        ###########################################################################################################################
        # SILENT SPLIT (NOT ALL random() KWARGS ARE NONE) #########################################################################
        elif int(np.sum(np.fromiter(map(lambda x: not x is None, (dev_count, dev_percent, test_count, test_percent)), dtype=np.int8))) == 1:
            # ONLY 1 KWARG IS NOT None

            _count, _percent = None, None
            # ANY DEV INDICATED, BUT NOT TEST
            if (not dev_count is None or not dev_percent is None) and (test_count is None and test_percent is None):
                if not dev_count is None: _count = dev_count
                elif not dev_percent is None: _percent = dev_percent
            # ANY TEST INDICATED BUT NOT DEV
            elif (not test_count is None or not test_percent is None) and (dev_count is None and dev_percent is None):
                if not test_count is None: _count = test_count
                elif not test_percent is None: _percent = test_percent

            if _count is None and _percent is None:
                raise self._exception(f'LOGIC IS FAILING IN DETERMINING WHETHER TO GET MASK SIZE FROM count OR percent', fxn)

            # KEEP _count OR CONVERT _percent TO _count TO KNOW HOW TO BUILD MASK
            if not _count is None:      #  SOMEWHAT ARBITRARY THAT TRAIN MUST HAVE AT LEAST 2
                if not self.bypass_validation: count_validator(_count, "", 1, self.rows - 2)
                # _count STAYS AS _count
            elif not _percent is None:      #  SOMEWHAT ARBITRARY THAT TRAIN MUST HAVE AT LEAST 2
                if not self.bypass_validation: percent_validator(_percent, "", 1, self.rows - 2)
                _count = int(np.floor(_percent/100 * self.rows))

            # MASK IS SIMPLY A [[len=self.rows]] OF RANDOM True/False, NUMBER OF Trues == dev_count OR test_count, WHICHEVER WAS GIVEN
            TRUES = nnrc.new_np_random_choice(range(self.rows), (1,_count), replace=False).reshape((1,-1))[0]
            self.MASK = np.zeros(self.rows, dtype=bool)   # FULL OF FALSES
            self.MASK[TRUES] = True
            del _count, TRUES
            self.MASK = self.MASK.reshape((1,-1)).astype(bool)

        else:   # 2 KWARGS ENTERED, 1 FROM dev, 1 FROM test, (CANT BE 2 FROM dev OR 2 FROM test)
            if not dev_count is None:    #  SOMEWHAT ARBITRARY THAT TRAIN MUST HAVE AT LEAST 2, BUT TEST MUST HAVE AT LEAST 1
                if not self.bypass_validation: count_validator(dev_count, 'dev', 1, self.rows-3)
                _count1 = int(dev_count)
            elif not dev_percent is None:    #  SOMEWHAT ARBITRARY THAT TRAIN MUST HAVE AT LEAST 2, BUT TEST MUST HAVE AT LEAST 1
                if not self.bypass_validation: percent_validator(dev_percent, 'dev', 1, self.rows-3)
                _count1 = int(np.floor(dev_percent/100 * self.rows))

            if not test_count is None:    #  SOMEWHAT ARBITRARY THAT TRAIN MUST HAVE AT LEAST 2
                if not self.bypass_validation: count_validator(test_count, 'test', 1, self.rows-2-_count1)
                _count2 = int(test_count)
            elif not test_percent is None:    #  SOMEWHAT ARBITRARY THAT TRAIN MUST HAVE AT LEAST 2
                if not self.bypass_validation: percent_validator(test_percent, 'test', 1, self.rows-2-_count1)
                _count2 = int(np.floor(test_percent/100 * self.rows))

            # EVEN THO VALIDATION SHOULD NOT ALLOW IT, JUST CHECK TO MAKE SURE THAT _count1 & _count2 TOTAL
            # LESS THAN self.rows AND THE ALLOWAND OF AT LEAST 2 FOR TRAIN
            if _count1 + _count2 > self.rows-2: raise self._exception(f'ROWS FOR DEV AND TEST ARE TOTALING TO A NUMBER '
                                f'GREATER THAN TOTAL ROWS MINUS 2 (REQUIRING AT LEAST 2 ROWS FOR TRAIN)', fxn)

            # DRAW RAND IDXS FOR TOTAL OF _count1 & _count2, THEN FROM THAT RANDOMLY DRAW THE TEST SET (AND WHAT IS LEFT
            # IS THE DEV SET)  CANT PULL SEPARATELY, CANT ALLOW THE CHANCE THAT SAME ROW IS PULLED TWICE
            DEV_AND_TEST_IDXS = nnrc.new_np_random_choice(range(self.rows), (1,_count1+_count2), replace=False).reshape((1,-1))[0]
            TEST_IDXS = nnrc.new_np_random_choice(DEV_AND_TEST_IDXS, (1,_count2), replace=False).reshape((1,-1))[0]
            # BUILD TEST_MASK FIRST (IS FINAL, AND NEEDED TO MAKE DEV_MASK)
            TEST_MASK = np.zeros(self.rows, dtype=bool)
            TEST_MASK[TEST_IDXS] = True
            del _count1, _count2, TEST_IDXS
            # BUILD DEV FROM DEV_AND_TEST_IDXS AND TEST_MASK
            DEV_MASK = np.zeros(self.rows, dtype=bool)
            DEV_MASK[DEV_AND_TEST_IDXS] = True
            DEV_MASK[TEST_MASK] = False
            del DEV_AND_TEST_IDXS

            # NOW JUST vstack DEV_MASK AND TEST_MASK TO GET A 2-COLUMN self.MASK
            self.MASK = np.vstack((DEV_MASK, TEST_MASK)).astype(bool)

        # END SILENT SPLIT (NOT ALL random() KWARGS ARE NONE) #####################################################################
        ###########################################################################################################################

        del approver, count_validator, percent_validator

        return self.core_run_fxn()


    def partition(self, number_of_partitions=None, dev_partition_number=None, test_partition_number=None):
        '''Generate Train-Test or Train-Dev-Test split by draw of specific partition.'''
        # SILENT (TWO OR THREE KWARGS PASSED) OR VERBOSE (NO KWARGS PASSED)
        # IF ANY _partition_number IS PASSED, MUST PASS number_of_partitions
        # IF ONLY DEV+nop OR TEST+nop PASSED: TWO-WAY PULL, IF nop+DEV+TEST PASSED: 3-WAY PULL
        # THE GOAL OF THIS MODULE IS TO GET THE INFORMATION TO BUILD A 1-COLUMN OR 2-COLUMN MASK, THEN THE MASK IS PASSED
        # TO core_run_fxn()

        fxn = inspect.stack()[0][3]

        def approver(func, ARGS):
            while True:
                output = func(*ARGS)
                if vui.validate_user_str(f'User entered {output}, Accept? (y/n) > ', 'YN') == 'Y': return output

        #####################################################################################################################################
        # IF ALL partition() KWARGS ARE NONE: VERBOSE SPLIT SETUP ##############################################################################
        if number_of_partitions is None and dev_partition_number is None and test_partition_number is None:

            row_text = f'({self.rows} rows)'
            get_partitions = lambda x: vui.validate_user_int(f'\nEnter number of partitions {row_text} > ',
                                                               min=2 if pull_setup == 'T' else 3, max=self.rows)
            get_partition_number = lambda name: vui.validate_user_int(f'\nEnter {name} partition number (zero-based) > ', min=0,
                                                               max=number_of_partitions-1)

            while True:
                pull_setup = vui.validate_user_str(f'Split TEST only(t) or split DEV and TEST(b) > ', 'TB')

                number_of_partitions = approver(get_partitions, ('',))

                if pull_setup == 'T':
                    dev_partition_number, test_partition_number = None, approver(get_partition_number, ('TEST',))
                elif pull_setup == 'B':
                    while True:
                        dev_partition_number = approver(get_partition_number, ('DEV',))
                        test_partition_number = approver(get_partition_number, ('TEST',))

                        if not dev_partition_number==test_partition_number: break
                        else: print(f'\n*** DEV PARTITION AND TEST PARTITION CANNOT BE THE SAME PARTITION ***\n')

                del pull_setup

                print(f'\nUser chose {number_of_partitions} partitions, '
                       + [f'partition {dev_partition_number} for dev, ' if not dev_partition_number is None else 'no dev partition, '][0]
                       + f'and partition {test_partition_number} for test')

                if vui.validate_user_str(f'Accept? (y/n) > ', 'YN') == 'Y':
                    del get_partitions, get_partition_number
                    break

            partition_size = int(np.floor(self.rows / number_of_partitions))

            # ONLY DO DEV IF 3 WAY SPLIT
            if not dev_partition_number is None:
                dev_part_start = dev_partition_number * partition_size
                dev_part_end = self.rows if dev_partition_number == number_of_partitions - 1 else (dev_partition_number + 1) * partition_size

            # MUST DO TEST METHOD WHETHER 2 OR 3 WAY SPLIT
            test_part_start = test_partition_number * partition_size
            test_part_end = self.rows if test_partition_number == number_of_partitions - 1 else (test_partition_number + 1) * partition_size

            if dev_partition_number is None:  # test_partition_number MUST NOT BE None
                self.MASK = np.zeros(self.rows, dtype=bool)
                self.MASK[test_part_start:test_part_end] = True

                self.MASK = self.MASK.reshape((1,-1)).astype(bool)

            elif not dev_partition_number is None:  # test_partition_number MUST NOT BE None
                self.MASK = np.zeros((2, self.rows), dtype=bool)
                self.MASK[0][dev_part_start:dev_part_end] = True
                self.MASK[1][test_part_start:test_part_end] = True
                del dev_partition_number, dev_part_start, dev_part_end
                self.MASK = self.MASK.astype(bool)

            del test_partition_number, test_part_start, test_part_end

        # END IF ALL partition() KWARGS ARE NONE: VERBOSE SPLIT SETUP #########################################################################
        #####################################################################################################################################

        ###########################################################################################################################
        # SILENT SPLIT (NOT ALL partition() KWARGS ARE NONE) #########################################################################
        else:
            if not self.bypass_validation:
                # IF PASSING KWARGS, CANT PASS ONE OR TWO, MUST PASS ZERO KWARGS OR MUST PASS number_of_partitions AND ONE OTHER

                if (not dev_partition_number is None or not test_partition_number is None) and number_of_partitions is None:
                    raise self._exception(f'IF A _partition_number IS INTENTIONALLY PASSED, MUST PASS number_of_partitions', fxn)

                if not number_of_partitions is None and (dev_partition_number is None and test_partition_number is None):
                    raise self._exception(f'IF number_of_partitions IS INTENTIONALLY PASSED, MUST ALSO PASS ONE OF '
                                          f'dev_partition_number OR test_partition_number OR BOTH', fxn)

                # SO number_of_partitions MUST HAVE BEEN PASSED
                if not number_of_partitions in range(self.rows):
                    raise self._exception(f'number_of_partitions ({number_of_partitions}) MUST BE AN INTEGER >= 1 AND <= {self.rows} ({self.rows} rows)', fxn)

                # dev_part OR test_part OR BOTH MUST HAVE BEEN PASSED
                if not dev_partition_number is None and not dev_partition_number in range(number_of_partitions):
                    raise self._exception(f'dev_partition_number ({dev_partition_number}) MUST BE AN INTEGER >= 0 AND <= number_of_partitions ({number_of_partitions}).')

                if not test_partition_number is None and not test_partition_number in range(number_of_partitions):
                    raise self._exception(f'test_partition_number ({test_partition_number}) MUST BE AN INTEGER >= 0 AND <= number_of_partitions ({number_of_partitions}).')

            partition_size = int(np.floor(self.rows / number_of_partitions))

            # IF ONLY MAKING 2-WAY SPLIT
            if int(np.sum(list(map(lambda x: not x is None, (dev_partition_number, test_partition_number))))) == 1:

                # VIA dev
                if not dev_partition_number is None and test_partition_number is None:
                    part_start = dev_partition_number * partition_size
                    part_end = self.rows if dev_partition_number == number_of_partitions - 1 else (dev_partition_number + 1) * partition_size

                # VIA test
                elif dev_partition_number is None and not test_partition_number is None:
                    part_start = test_partition_number * partition_size
                    part_end = self.rows if test_partition_number == number_of_partitions - 1 else (test_partition_number + 1) * partition_size

                self.MASK = np.zeros(self.rows, dtype=bool)
                self.MASK[np.fromiter(range(part_start, part_end), dtype=np.int32)] = True
                self.MASK = self.MASK.reshape((1,-1))

            # IF MAKING 3-WAY SPLIT, VIA dev AND test
            elif not dev_partition_number is None and not test_partition_number is None:

                # dev_part AND test_part CANNOT BE THE SAME PARTITION
                if dev_partition_number == test_partition_number:
                    raise self._exception(f'dev_partition_number AND test_partition_number CANNOT BE THE SAME.')

                dev_part_start = dev_partition_number * partition_size
                test_part_start = test_partition_number * partition_size

                # IF number_of_partitions DOES NOT DIVIDE EVENLY INTO len(DATA), THEN THERE WILL BE (len(DATA) % partitions)
                # NUMBER OF DATA ROWS AT THE BOTTOM THAT WOULD NEVER BE INCLUDED IN DEV OR TEST SET, SO INCLUDE THEM ONLY IN LAST PARTITION
                dev_part_end = self.rows if dev_partition_number == number_of_partitions - 1 else (dev_partition_number + 1) * partition_size
                test_part_end = self.rows if test_partition_number == number_of_partitions - 1 else (test_partition_number + 1) * partition_size

                self.MASK = np.zeros((2, self.rows), dtype=bool)
                self.MASK[0][np.fromiter(range(dev_part_start, dev_part_end), dtype=np.int32)] = True
                self.MASK[1][np.fromiter(range(test_part_start, test_part_end), dtype=np.int32)] = True

                self.MASK = self.MASK.astype(bool)

        # END SILENT SPLIT (NOT ALL partition() KWARGS ARE NONE) #########################################################################
        ###########################################################################################################################

        del approver

        return self.core_run_fxn()


    def category(self, object_name_for_dev=None, dev_column_name_or_idx=None, DEV_SPLIT_ON_VALUES_AS_LIST=None,
                     object_name_for_test=None, test_column_name_or_idx=None, TEST_SPLIT_ON_VALUES_AS_LIST=None,
                     DATA_FULL_SUPOBJ=None, TARGET_FULL_SUPOBJ=None, REFVECS_FULL_SUPOBJ=None):

        '''select examples for dev & test sets using categories in the objects.'''

        # MLRunTemplate CALLS ON self.SWNL TO GET OBJS & HEADERS, CALLS ON self.VAL_DTYPES & self.MOD_DTYPES
        # BUT TRAIN_SWNL MUST BE PASSED FOR SPLIT, BECAUSE TEST IS GOT AFTER DEV, AND DEV CHANGED SWNL INTO TRAIN

        # LAYOUT
        #  I) VALIDATE KWARGS / DETERMINE SILENT OR VERBOSE BASED ON KWARGS
        #    A) VALIDATE FULL_SUPOBJS AGAINST OBJECTS (COULD BE PASSED AND USED FOR SILENT MODE (COLUMN PICK) OR VERBOSE (PRINT COLUMNS))
        #    B) VALIDATE SILENT-MODE KWARGS AND DETERMINE IF SILENT OR VERBOSE MODE, AND IF SILENT IS IT TEST ONLY OR DEV & TEST
        # II) VERBOSE (x_obj_name, x_col_idx, x_split_on NOT NEEDED, AVAILABLE SUPOBJS PASSED, OR MAKE DUMMIES IF NO SUPOBJS)
        #    A) PROMPT USER TO CREATE TEST ONLY OR DEV / TEST
        #       Loop over ['BUILD_TEST'] OR ['BUILD_DEV','BUILD_TEST']:
        #       B)  PRINT A DISPLAY OF ALL OBJECTS HEADERS & MOD_DTYPES
        #           1) IF SUPOBJ AVAILABLE CREATE MENU FROM IT
        #           2) IF NO SUPOBJ, OPTIONALLY USE bfso TO GET THEM
        #       C)  USER SELECTS OBJECT, THEN COLUMN IDX.
        #       D)  GETS UNIQUES IN THAT OBJ/COL
        #       E)  USER SELECTS VALUES THAT GO INTO DEV OR TEST & DELETED FROM TRAIN
        #           1) SELECT VALUES THAT SIGNIFY DEV OR TEST, AND ACCEPT SELECTIONS
        #       F)  EACH LOOP CREATES A SINGLE MASK VECTOR THAT IS BOOL INDICATING THAT ROW GOES INTO THAT LOOP'S OBJECT.
        #    G) IF GETTING DEV & TEST, ENSURE NO ROW HAS TWO Trues, OTHERWISE FORCE RESTART
        #
        #III) SILENT MODE (obj_name_for_dev, dev_col_idx, DEV_SPLIT_ON, obj_name_for_test, test_col_idx, TEST_SPLIT_ON ARE PASSED)
        #    A) BUILD MASK THAT IS BOOL, ONE COLUMN FOR TEST ONLY, 2 COLUMNS FOR DEV-TEST
        #
        # IV) APPLY MASK TO DATA VIA core_run_fxn() WHICH SILENTLY DELETES MASKED ROWS FROM
        #     DATA AND RETURNS AN OLD-FASHIONED SWNL TYPE OBJECT WHICH IS ASSIGNED TO DEV_SWNL OR TEST_SWNL

        fxn = inspect.stack()[0][3]

        ##############################################################################################################################
        #  I) VALIDATE KWARGS / DETERMINE SILENT OR VERBOSE BASED ON KWARGS ##########################################################
        #       A) VALIDATE FULL_SUPOBJS AGAINST OBJECTS (COULD BE PASSED AND USED FOR SILENT MODE (COLUMN PICK) OR VERBOSE (PRINT COLUMNS))

        if not DATA_FULL_SUPOBJ is None and not self.DATA is None:
            vfso.validate_full_support_object(DATA_FULL_SUPOBJ, OBJECT=self.DATA,
              object_given_orientation=self.data_given_orientation, OBJECT_HEADER=None, allow_override=False)
        if not TARGET_FULL_SUPOBJ is None and not self.TARGET is None:
            vfso.validate_full_support_object(TARGET_FULL_SUPOBJ, OBJECT=self.TARGET,
              object_given_orientation=self.target_given_orientation, OBJECT_HEADER=None, allow_override=False)
        if not REFVECS_FULL_SUPOBJ is None and not self.REFVECS is None :
            vfso.validate_full_support_object(REFVECS_FULL_SUPOBJ, OBJECT=self.REFVECS,
              object_given_orientation=self.refvecs_given_orientation, OBJECT_HEADER=None, allow_override=False)
        #        END A) VALIDATE FULL_SUPOBJS AGAINST OBJECTS (COULD BE PASSED AND USED FOR SILENT MODE (COLUMN PICK) OR VERBOSE (PRINT COLUMNS))

        #        B) VALIDATE SILENT-MODE KWARGS AND DETERMINE IF SILENT OR VERBOSE MODE, AND IF SILENT IS IT TEST ONLY OR DEV & TEST #########

        COL_DICT = {'DATA': self.data_cols, 'TARGET': self.target_cols, 'REFVECS': self.refvec_cols}
        OBJ_DICT = {'DATA': self.DATA, 'TARGET': self.TARGET, 'REFVECS': self.REFVECS}
        ORIENT_DICT = {'DATA': self.data_given_orientation, 'TARGET': self.target_given_orientation, 'REFVECS': self.refvecs_given_orientation}
        SUPOBJ_DICT = {'DATA': DATA_FULL_SUPOBJ, 'TARGET': TARGET_FULL_SUPOBJ, 'REFVECS': REFVECS_FULL_SUPOBJ}

        silent_dev, silent_test = False, False

        DEV_KWARG_SET = (object_name_for_dev, dev_column_name_or_idx, DEV_SPLIT_ON_VALUES_AS_LIST)
        TEST_KWARG_SET = (object_name_for_test, test_column_name_or_idx, TEST_SPLIT_ON_VALUES_AS_LIST)

        DEV_COLUMN_IN_QUESTION, TEST_COLUMN_IN_QUESTION = None, None

        if not self.bypass_validation:
            if True in map(lambda x: x is None, DEV_KWARG_SET) and True in map(lambda y: not y is None, DEV_KWARG_SET):
                raise self._exception(f'IF ANY DEV KWARGS ARE PASSED, ALL MUST BE PASSED (OBJECT, COLUMN, AND SPLIT_ON_VALUES)', fxn=fxn)

            if True in map(lambda x: x is None, TEST_KWARG_SET) and True in map(lambda y: not y is None, TEST_KWARG_SET):
                raise self._exception(f'IF ANY TEST KWARGS ARE PASSED, ALL MUST BE PASSED (OBJECT, COLUMN, AND SPLIT_ON_VALUES)', fxn=fxn)

            # DEFINE A FUNCTION TO HANDLE VALIDATION FOR DEV AND TEST
            def silent_kwarg_validator_for_dev_or_test(name, object_name_for_x, x_column_name_or_idx, X_SPLIT_ON_VALUES_AS_LIST):
                ALLOWED_OBJ_NAMES = ['DATA', 'TARGET', 'REFVECS']

                # object_name_for_x
                if object_name_for_x not in ALLOWED_OBJ_NAMES: raise self._exception(
                    f'object_name_for_{name.lower()} must be in {", ".join(ALLOWED_OBJ_NAMES)}.', fxn=fxn)

                # x_column_name_or_idx
                _, __ = object_name_for_x, x_column_name_or_idx
                if 'INT' in str(type(__)).upper():
                    if not __ in range(-COL_DICT[_], COL_DICT[_]-1):
                        raise self._exception(f'{name.lower()}_column_name_or_idx "{__}" IS OUT OF RANGE FOR {_} [{-COL_DICT[_]}, {COL_DICT[_]-1}]', fxn=fxn)
                    x_col_idx = x_column_name_or_idx
                elif isinstance(__, str) and SUPOBJ_DICT[_] is None:
                    raise self._exception(f'{name.upper()} COLUMN CAN ONLY BE CHOSEN BY COLUMN NAME IF RESPECTIVE SUPOBJ IS PASSED.', fxn=fxn)
                elif isinstance(__, str) and not SUPOBJ_DICT[_] is None:
                    if __ not in SUPOBJ_DICT[_][msod.master_support_object_dict()['HEADER']['position']]:
                        raise self._exception(f'PASSED {name.lower()}_column_name_or_idx "{__}" IS NOT IN {_} COLUMNS', fxn=fxn)
                    x_col_idx = np.argwhere(SUPOBJ_DICT[_][msod.master_support_object_dict()['HEADER']['position']], __)[-1].reshape((1,-1))[0][0]
                else:
                    raise self._exception(f'INVALID TYPE "{type(__)}" PASSED FOR {name.lower()}_column_name_or_idx. MUST BE int OR str.')

                # X_SPLIT_ON_VALUES_AS_LIST
                if not isinstance(X_SPLIT_ON_VALUES_AS_LIST, (np.ndarray, list, tuple)):
                    raise self._exception(f'{name.upper()}_SPLIT_ON_VALUES_AS_LIST MUST BE PASSED AS A LIST-TYPE', fxn=fxn)

                # REMEMBER THE COLUMN BEING SPLIT ON COULD BE IN A SPARSE DICT, SO GET THE COLUMN OUT AND INTO NP FORMAT
                ColumnInQuestion = mlrco.MLRowColumnOperations(OBJ_DICT[_], ORIENT_DICT[_], name=_, bypass_validation=True)
                COLUMN_IN_QUESION = ColumnInQuestion.return_columns([x_col_idx], return_orientation='COLUMN', return_format='ARRAY')
                UNIQUES = np.unique(COLUMN_IN_QUESION.reshape((1, -1)))
                del ColumnInQuestion

                for value in X_SPLIT_ON_VALUES_AS_LIST:
                    if value not in UNIQUES:
                        raise self._exception(f'{name.upper()}_SPLIT_ON_VALUE "{value}" IS NOT IN INDICATED COLUMN')

                del ALLOWED_OBJ_NAMES, _, __, x_col_idx, UNIQUES, x_column_name_or_idx, object_name_for_x

                return COLUMN_IN_QUESION

        # IF DEV KWARGS PASSED, INDICATE DOING SILENT DEV & OPTIONALLY VALIDATE
        if True not in map(lambda x: x is None, DEV_KWARG_SET):
            if not self.bypass_validation:
                DEV_COLUMN_IN_QUESTION = silent_kwarg_validator_for_dev_or_test('DEV', object_name_for_dev,
                                            dev_column_name_or_idx, DEV_SPLIT_ON_VALUES_AS_LIST)
            silent_dev = True

        # IF TEST KWARGS PASSED, INDICATE DOING SILENT TEST & OPTIONALLY VALIDATE
        if True not in map(lambda x: x is None, TEST_KWARG_SET):
            if not self.bypass_validation:
                TEST_COLUMN_IN_QUESTION = silent_kwarg_validator_for_dev_or_test('TEST', object_name_for_test,
                                            test_column_name_or_idx, TEST_SPLIT_ON_VALUES_AS_LIST)
            silent_test = True

        del DEV_KWARG_SET, TEST_KWARG_SET, silent_kwarg_validator_for_dev_or_test

        # IF ONLY ONE OF dev OR silent IS BEING DONE, JUST CALL IT test; IF BOTH, KEEP dev & test; IF ZERO IS VERBOSE
        silent = int(silent_dev) + int(silent_test)

        if silent == 1:
            silent_dev, silent_test = False, True
            # IF INFO IS UNDER dev KWARGS (test ARE None), MOVE THOSE KWARGS TO test & SET dev TO None
            if object_name_for_test is None:   # IMPLIES test_column_name_or_idx, TEST_SPLIT_ON_VALUES_AS_LIST ARE ALSO None
                object_name_for_test = object_name_for_dev; object_name_for_dev = None
                test_column_name_or_idx = dev_column_name_or_idx; dev_column_name_or_idx = None
                TEST_SPLIT_ON_VALUES_AS_LIST = DEV_SPLIT_ON_VALUES_AS_LIST; DEV_SPLIT_ON_VALUES_AS_LIST = None
                TEST_COLUMN_IN_QUESTION = DEV_COLUMN_IN_QUESTION; DEV_COLUMN_IN_QUESTION = None

        elif silent == 2:
            silent_dev, silent_test = True, True
        #  END    B) VALIDATE SILENT-MODE KWARGS AND DETERMINE IF SILENT OR VERBOSE MODE, AND IF SILENT IS IT TEST ONLY OR DEV & TEST #########

        #  END I) VALIDATE KWARGS / DETERMINE SILENT OR VERBOSE BASED ON KWARGS ###################################################
        ##############################################################################################################################


        ##############################################################################################################################
        # II) VERBOSE (x_obj_name, x_col_idx, x_split_on NOT NEEDED, AVAILABLE SUPOBJS PASSED, OR MAKE DUMMIES IF NO SUPOBJS) #########

        if not silent:

            while True:  # TO ALLOW RESTART / ABORT.  return EXITS THE while, NOT break
        #       A) PROMPT USER TO CREATE TEST ONLY OR DEV / TEST ###############################################################
                while True:
                    pull_setup = vui.validate_user_str(f'Split TEST only(t) or split DEV and TEST(b) > ', 'TB')
                    if vui.validate_user_str(f"User selected {dict((('T','TEST only'),('B','DEV and TEST')))[pull_setup]}. Accept? (y/n) > ", 'YN') == 'Y': break

                if pull_setup == 'T': verbose_dev, verbose_test = False, True
                elif pull_setup == 'B': verbose_dev, verbose_test = True, True
                del pull_setup
        #       END A) PROMPT USER TO CREATE TEST ONLY OR DEV / TEST ###############################################################

                mask_size = int(verbose_dev) + int(verbose_test)
                self.MASK = np.empty((mask_size, self.rows), dtype=bool)
        #       Loop over ['BUILD_TEST'] OR ['BUILD_DEV','BUILD_TEST']
                for mask_idx in range(mask_size):
                    loop_name = ["DEV", "TEST"][mask_idx] if mask_size==2 else "TEST"
            #       B)  PRINT A DISPLAY OF ALL OBJECTS HEADERS & MOD_DTYPES
                    #   1) IF SUPOBJ AVAILABLE CREATE MENU FROM IT
                    #   2) IF NO SUPOBJ, OPTIONALLY USE bfso TO GET THEM

                    #### PRINT COLUMN NAMES AND DTYPES ###########################################################################

                    print(f'\nSELECT COLUMN FOR {loop_name} SPLIT:')

                    if 'DATA' in self.ACTV_NAMES:
                        print()
                        print(f'DATA:')
                        sop.GeneralSmallObjectPreview(
                                                    self.DATA,
                                                    self.data_given_orientation,
                                                    SINGLE_OR_FULL_SUPPORT_OBJECT=DATA_FULL_SUPOBJ,
                                                    support_name='MODIFIEDDATATYPES',
                                                    idx=None)

                    if 'TARGET' in self.ACTV_NAMES:    # NOT elif!
                        print()
                        print(f'TARGET:')
                        sop.GeneralSmallObjectPreview(
                                                    self.TARGET,
                                                    self.target_given_orientation,
                                                    SINGLE_OR_FULL_SUPPORT_OBJECT=TARGET_FULL_SUPOBJ,
                                                    support_name='MODIFIEDDATATYPES',
                                                    idx=None)

                    if 'REFVECS' in self.ACTV_NAMES:    # NOT elif!
                        print()
                        print(f'REFVECS:')
                        sop.GeneralSmallObjectPreview(
                                                    self.REFVECS,
                                                    self.refvecs_given_orientation,
                                                    SINGLE_OR_FULL_SUPPORT_OBJECT=REFVECS_FULL_SUPOBJ,
                                                    support_name='MODIFIEDDATATYPES',
                                                    idx=None)

                    '''
                    OLD PRINT LAYOUT --- KEEP IN CASE WANT TO GO BACK 3/28/23 -- THIS WAS PULLED FROM MLRunTemplate
                    print()
                    print(f' '.ljust(70) + f'VALIDATED'.ljust(12) + f'MODIFIED'.ljust(12))
                    print(f'COLUMN NAME'.ljust(70) + f'DATATYPE'.ljust(12) + f'DATATYPE'.ljust(12))
                    for __ in self.OBJ_IDXS:
                        print("WORKING " + str(self.SUPER_NUMPY_DICT[__]))
                        for x in range(len(self.SUPER_WORKING_NUMPY_LIST[__ + 1][0])):
                            aaa = self.SUPER_WORKING_NUMPY_LIST[__ + 1][0][x][:65]
                            bbb = self.WORKING_VALIDATED_DATATYPES[__][x]
                            ccc = self.WORKING_MODIFIED_DATATYPES[__][x]
                            print(f'{aaa}'.ljust(70) + f'{bbb}'.ljust(12) + f'{ccc}'.ljust(12))
                        print()
                    '''
                    #### B) END PRINT COLUMN NAMES AND DTYPES ########################################################################

                #   C)  USER SELECTS OBJECT, THEN COLUMN IDX.
                    #### USER SELECT OBJECT AND COLUMN TO USE IN SELECTING DEV/TEST SET ###########################################
                    while True:
                        obj_idx = ls.list_single_select(self.ACTV_NAMES, f'Select object with the column to do {loop_name} split on', 'idx')[0]

                        ACTV_SUPOBJ = SUPOBJ_DICT[self.ACTV_NAMES[obj_idx]]

                        if not ACTV_SUPOBJ is None:
                            col_idx = ls.list_single_select(ACTV_SUPOBJ[msod.master_support_object_dict()["HEADER"]["position"]],
                                                            f'\nSelect column to split on', 'idx')[0]
                            col_name = ACTV_SUPOBJ[msod.master_support_object_dict()["HEADER"]["position"]][col_idx]
                        elif ACTV_SUPOBJ is None:
                            col_idx = vui.validate_user_int(f'Select column index to split on > ', min=0,
                                max={'DATA':self.data_cols,'TARGET':self.target_cols,'REFVECS':self.refvec_cols}[self.ACTV_NAMES[obj_idx]]-1)
                            col_name = f'COLUMN{col_idx+1}'

                        del ACTV_SUPOBJ

                        if vui.validate_user_str(f'\nUser selected {self.ACTV_NAMES[obj_idx]}, {col_name}. Accept? (y/n) > ', 'YN') == 'Y': break

                    #### END USER SELECTS OBJECT AND COLUMN TO USE IN SELECTING DEV/TEST SET ###########################################

            #       D)  GETS UNIQUES IN THAT OBJ/COL
                    #### GET UNIQUES IN USER SELECTED OBJ/COL ########################################################################
                    UNIQUES = mlo.MLObject(self.ACTV_OBJECTS[obj_idx],
                                           self.ACTV_ORIENTATIONS[obj_idx],
                                           name=self.ACTV_NAMES[obj_idx],
                                           return_orientation='AS_GIVEN',
                                           return_format='AS_GIVEN',
                                           bypass_validation=self.bypass_validation,
                                           calling_module=self.this_module,
                                           calling_fxn=fxn).unique(col_idx)

                    #### END GET UNIQUES IN USER SELECTED OBJ/COL ####################################################################

                    #### HANDLE SELECTION OF VALUES USED TO BUILD DEV/TEST AND DELETED FROM TRAIN ################################
            #       E)  USER SELECTS VALUES THAT GO INTO DEV OR TEST & DELETED FROM TRAIN
                    print(f'\nSelect categories that will be used to build {["DEV", "TEST"][mask_idx]} set and deleted from TRAIN:')
            #           1) SELECT VALUES THAT SIGNIFY DEV OR TEST, AND ACCEPT SELECTIONS
                    while True:
                        DEV_OR_TEST_CATEGORIES = ls.list_custom_select(UNIQUES, 'value')
                        print(f'\nUser selected to build {loop_name} objects from column "{col_name}" in {self.ACTV_NAMES[obj_idx]} '
                              f'using value(s): {", ".join(map(str, DEV_OR_TEST_CATEGORIES))}.')
                        if vui.validate_user_str(f'Accept? (y/n) > ', 'YN') == 'Y': del UNIQUES; break

                    #### END HANDLE SELECTION OF CATEGORIES USED TO BUILD DEV/TEST AND DELETED FROM TRAIN #############################

            #       F)  EACH LOOP CREATES A SINGLE MASK VECTOR THAT IS BOOL INDICATING THAT ROW GOES INTO THAT LOOP'S OBJECT.
                    # BUILD A MASK TO KNOW WHAT ROWS TO MOVE TO DEV / TEST OBJECT AND DELETE FROM TRAIN #####################################

                    self.MASK[mask_idx] = np.fromiter((_ in DEV_OR_TEST_CATEGORIES for _ in self.ACTV_OBJECTS[obj_idx][col_idx]),
                                            dtype=bool).reshape((1, -1))[0]

                    del obj_idx, col_idx, col_name, DEV_OR_TEST_CATEGORIES

            #   G) IF GETTING DEV & TEST, ENSURE NO ROW HAS TWO Trues, OTHERWISE FORCE RESTART
                ROW_SUMS = np.sum(self.MASK, axis=0)
                if not (min(ROW_SUMS)==0 and max(ROW_SUMS)==1):
                    if min(ROW_SUMS)==0 and max(ROW_SUMS)==0:
                        print(f'\n*** ERROR IN MASK. ALL ROWS SUM TO ZERO, MEANING NO MASK WILL BE APPLIED. ***\n')
                    if max(ROW_SUMS) > 1:
                        print(f'\n*** ERROR IN MASK. AT LEAST ONE ROW SUMS TO GREATER THAN 1. ***\n')
                    if min(ROW_SUMS) == 1:
                        print(f'\n*** ERROR IN MASK. ALL ROWS SUMS TO AT LEAST ONE, INDICATING EVERYTHING WILL MOVE OUT OF TRAIN. ***\n')

                    continue # FORCE BACK TO TOP OF OUTERMOST while

                del mask_size, loop_name, ROW_SUMS
                break

        # END II) VERBOSE (x_obj_name, x_col_idx, x_split_on NOT NEEDED, AVAILABLE SUPOBJS PASSED, OR MAKE DUMMIES IF NO SUPOBJS) #########
        ##############################################################################################################################

        #III) SILENT MODE (obj_name_for_dev, dev_col_idx, DEV_SPLIT_ON, obj_name_for_test, test_col_idx, TEST_SPLIT_ON ARE PASSED)
        elif silent:
        #     A) BUILD MASK THAT IS BOOL, ONE COLUMN FOR TEST ONLY, 2 COLUMNS FOR DEV-TEST ########################################
            # TEST COLUMN IS NEED WHETHER SPLITTING TEST ONLY OR DEV & TEST

            if not TEST_COLUMN_IN_QUESTION is None:
                # IF not self.bypass_validation, THE COLUMN WAS ALREADY CREATED, JUST USE THAT COLUMN FROM BEFORE
                pass
            elif TEST_COLUMN_IN_QUESTION is None:   # MUST GET COLUMN IN QUESTION OUT OF PERTINENT OBJECT
                ColumnInQuestion = mlrco.MLRowColumnOperations(
                                                                OBJ_DICT[object_name_for_test],
                                                                ORIENT_DICT[object_name_for_test],
                                                                name=object_name_for_test,
                                                                bypass_validation=self.bypass_validation)

                if 'INT' in str(type(test_column_name_or_idx)).upper():
                    x_col_idx = test_column_name_or_idx
                else:
                    x_col_idx = np.argwhere(SUPOBJ_DICT[object_name_for_test][msod.master_support_object_dict()['HEADER']['position']], test_column_name_or_idx)[-1].reshape((1,-1))[0][0]

                TEST_COLUMN_IN_QUESTION = ColumnInQuestion.return_columns([x_col_idx], return_orientation='COLUMN',
                                                                return_format='ARRAY')
                del ColumnInQuestion, x_col_idx

            # CONVERT IT TO BOOLS USING TEST_SPLIT_ON_VALUES_AS_LIST
            DUM_TEST_COLUMN_IN_QUESTION = np.zeros((1,self.rows), dtype=bool)
            for test_value in TEST_SPLIT_ON_VALUES_AS_LIST:
                DUM_TEST_COLUMN_IN_QUESTION[..., TEST_COLUMN_IN_QUESTION==test_value] = True
            TEST_COLUMN_IN_QUESTION = DUM_TEST_COLUMN_IN_QUESTION; del DUM_TEST_COLUMN_IN_QUESTION

            if not silent_dev:
                self.MASK = TEST_COLUMN_IN_QUESTION.reshape((1,-1))

            if silent_dev and silent_test:  # CREATE DEV IF NEEDED
                if not DEV_COLUMN_IN_QUESTION is None:
                    # IF not self.bypass_validation, THE COLUMN WAS ALREADY CREATED, JUST USE THAT COLUMN FROM BEFORE
                    pass
                elif DEV_COLUMN_IN_QUESTION is None:  # MUST GET COLUMN IN QUESTION OUT OF PERTINENT OBJECT
                    ColumnInQuestion = mlrco.MLRowColumnOperations(
                                OBJ_DICT[object_name_for_dev],
                                ORIENT_DICT[object_name_for_dev],
                                name=object_name_for_dev,
                                bypass_validation=self.bypass_validation)

                    if 'INT' in str(type(dev_column_name_or_idx)).upper():
                        x_col_idx = dev_column_name_or_idx
                    else:
                        x_col_idx = np.argwhere(SUPOBJ_DICT[object_name_for_dev][msod.master_support_object_dict()['HEADER']['position']], dev_column_name_or_idx)[-1].reshape((1, -1))[0][0]
                    DEV_COLUMN_IN_QUESTION = ColumnInQuestion.return_columns([x_col_idx], return_orientation='COLUMN',
                                                                             return_format='ARRAY')
                    del ColumnInQuestion
                # CONVERT IT TO BOOLS USING DEV_SPLIT_ON_VALUES_AS_LIST
                DUM_DEV_COLUMN_IN_QUESTION = np.zeros((1,self.rows), dtype=bool)
                for dev_value in DEV_SPLIT_ON_VALUES_AS_LIST:
                    DUM_DEV_COLUMN_IN_QUESTION[..., DEV_COLUMN_IN_QUESTION==dev_value] = True
                DEV_COLUMN_IN_QUESTION = DUM_DEV_COLUMN_IN_QUESTION; del DUM_DEV_COLUMN_IN_QUESTION
                # DEV_COLUMN_IN_QUESTION = np.fromiter((_ in DEV_SPLIT_ON_VALUES_AS_LIST for _ in DEV_COLUMN_IN_QUESTION), dtype=bool)

                self.MASK = np.vstack((DEV_COLUMN_IN_QUESTION, TEST_COLUMN_IN_QUESTION))

                del DEV_COLUMN_IN_QUESTION, TEST_COLUMN_IN_QUESTION

                if not self.bypass_validation:  # CHECK TO SEE IF DOUBLE MASK ONLY SUMS TO ZEROS & ONES FOR ALL ROWS
                    ROW_SUMS = np.sum(self.MASK, axis=0)
                    if not (min(ROW_SUMS)==0 and max(ROW_SUMS)==1):
                        raise self._exception(f'MASK ROWS MUST SUM TO ZEROS AND ONES. CANNOT BE ALL ZEROS OR ALL ONES.', fxn=fxn)
                    del ROW_SUMS

        #    END A) BUILD MASK THAT IS BOOL, ONE COLUMN FOR TEST ONLY, 2 COLUMNS FOR DEV-TEST ########################################

        del COL_DICT, OBJ_DICT, ORIENT_DICT, SUPOBJ_DICT, silent_dev, silent_test
        # END III) SILENT MODE (obj_name_for_dev, dev_col_idx, DEV_SPLIT_ON, obj_name_for_test, test_col_idx, TEST_SPLIT_ON ARE PASSED)

        # IV) APPLY MASK TO ALL OBJECTS VIA core_run_fxn()
        self.core_run_fxn()


    # OVERWRITES
    def core_run_fxn(self):

        # LEAVE train_dev_test_split_core() SEPARATE, IT IS CONVENIENT TO TEST IN THE CURRENT ARRANGEMENT

        # self.TRAIN, self.DEV, self.TEST ARE RETURNED, BUT self.DATA, self.TARGET, self.REFVECS CAN BE ACCESSED AS ATTRS

        # self.DATA, self.TARGET, self.REFVECS LOOK LIKE:
        # 2 WAY SPLIT, EG FOR DATA:  (DATA_TRAIN, DATA_TEST)
        # 3 WAY SPLIT, EG FOR TARGET:  (TARGET_TRAIN, TARGET_DEV, TARGET_TEST)

        # self.TRAIN, self.DEV, self.TEST LOOK LIKE:
        # FOR 2 OBJECTS (DATA & TARGET IN THIS CASE), EG self.TRAIN:  (DATA_TRAIN, TARGET_TRAIN)
        # FOR 3 OBJECTS, EG self.DEV:  (DATA_DEV, TARGET_DEV, REFVECS_DEV)

        # self.TRAIN, OPTIONALLY self.DEV, AND self.TEST ARE return
        # 2 WAY SPLIT OF 2 OBJECTS RETURNS, EG:  (DATA_TRAIN, TARGET_TRAIN), (DATA_TEST, TARGET_TEST)
        # 3 WAY SPLIT OF 3 OBJECTS RETURNS, EG:  (DATA_TRAIN, TARGET_TRAIN, REFVECS_TRAIN), (DATA_DEV, TARGET_DEV, REFVECS_DEV), (DATA_TEST, TARGET_TEST, REFVECS_TEST)


        RETURN_TUPLE = tuple()

        # train_dev_test_split_core() RETURNS (TRAIN_OBJ1, TEST_OBJ1) OR (TRAIN_OBJ1, DEV_OBJ1, TEST_OBJ1)
        for name, _OBJECT, _given_orientation in zip(self.ACTV_NAMES, self.ACTV_OBJECTS, self.ACTV_ORIENTATIONS):
            RETURN_TUPLE += tdtsc.train_dev_test_split_core(_OBJECT, _given_orientation, self.MASK, self.bypass_validation)

        # HOW TO HASH OUT WHAT RETURN_TUPLE LOOKS LIKE, INSTEAD OF GETTING AND SENDING ALL THAT INFORMATION OUT OF EACH OF THE METHODS?
        # ANY OF THE "self.ACTV"s WILL GIVE NUMBER OF OBJECTS SPLIT, SO TOTAL OBJECTS RETURNED DIVIDED BY len(ACTV) INDICATES IF WAS
        # TRAIN-TEST OR TRAIN-DEV-TEST
        number_of_objects_split = len(self.ACTV_NAMES)
        number_in_split = int(len(RETURN_TUPLE) / number_of_objects_split)

        # YIELD TUPLES OF (TRAIN, TEST) OR (TRAIN, DEV, TEST) FOR A DATA, TARGET, OR REFVEC OBJECT
        self.DATA = RETURN_TUPLE[:number_in_split] if 'DATA' in self.ACTV_NAMES else None
        if not 'TARGET' in self.ACTV_NAMES: self.TARGET = None
        elif not 'DATA' in self.ACTV_NAMES: self.TARGET = RETURN_TUPLE[:number_in_split]
        else: self.TARGET = RETURN_TUPLE[number_in_split:2*number_in_split]
        self.REFVECS = RETURN_TUPLE[-number_in_split:] if 'REFVECS' in self.ACTV_NAMES else None

        # YIELD TUPLES OF (TRAIN,TRAIN,..),(TEST,TEST,...) OR (TRAIN, TRAIN,..),(DEV,DEV,..),(TEST,TEST,...)
        self.TRAIN = tuple(RETURN_TUPLE[_] for _ in range(0, number_of_objects_split*number_in_split, number_in_split))
        self.DEV = None if number_in_split == 2 else tuple(RETURN_TUPLE[_] for _ in range(1, number_of_objects_split*number_in_split, number_in_split))
        self.TEST = tuple(RETURN_TUPLE[_] for _ in range(1, number_of_objects_split * number_in_split, number_in_split)) if number_in_split == 2 else \
            tuple(RETURN_TUPLE[_] for _ in range(2, number_of_objects_split * number_in_split, number_in_split))


        return (self.TRAIN, self.TEST) if number_in_split == 2 else (self.TRAIN, self.DEV, self.TEST)



























if __name__ == '__main__':
    # TEST MODULE

    # FUNCTIONAL CODE AND TEST CODE VERIFIED GOOD 4/3/23

    ########################################################################################################################
    # TEST mask() ##########################################################################################################
    print(f'\033[92mSTARTING mask() TESTS...\033[0m')

    # ALREADY KNOW THAT SPLITS CORRECTLY FROM train_dev_test_split_core() TESTS.
    # ESTABLISH THAT core_return_fxn() RETURNS self.DATA, self.TARGET, self.REFVECS, self.TRAIN, self.DEV, & self.TEST CORRECTLY.

    DATA = np.random.randint(0, 10, (3, 5))
    TARGET = np.random.randint(0, 10, (3, 5))
    REFVECS = np.random.randint(0, 10, (3, 5))
    _orient = 'COLUMN'
    MASK = [[False, False, False, False, True], [False, False, False, True, False]]

    DATA_TRAIN = np.delete(DATA, np.sum(MASK, axis=0).astype(bool), axis=1)
    DATA_DEV = DATA[...,MASK[0]]
    DATA_TEST = DATA[..., MASK[1]]
    TARGET_TRAIN = np.delete(TARGET, np.sum(MASK, axis=0).astype(bool), axis=1)
    TARGET_DEV = TARGET[...,MASK[0]]
    TARGET_TEST = TARGET[..., MASK[1]]
    REFVECS_TRAIN = np.delete(REFVECS, np.sum(MASK, axis=0).astype(bool), axis=1)
    REFVECS_DEV = REFVECS[...,MASK[0]]
    REFVECS_TEST = REFVECS[..., MASK[1]]

    EXP_DATA = (DATA_TRAIN, DATA_DEV, DATA_TEST)
    EXP_TARGET = (TARGET_TRAIN, TARGET_DEV, TARGET_TEST)
    EXP_REFVECS = (REFVECS_TRAIN, REFVECS_DEV, REFVECS_TEST)
    EXP_TRAIN = (DATA_TRAIN, TARGET_TRAIN, REFVECS_TRAIN)
    EXP_DEV = (DATA_DEV, TARGET_DEV, REFVECS_DEV)
    EXP_TEST = (DATA_TEST, TARGET_TEST, REFVECS_TEST)

    TestClass = TrainDevTestSplit(DATA=DATA,
                                  TARGET=TARGET,
                                  REFVECS=REFVECS,
                                  data_given_orientation=_orient,
                                  target_given_orientation=_orient,
                                  refvecs_given_orientation=_orient,
                                  bypass_validation=False)

    TestClass.mask(MASK)

    ACT_DATA = TestClass.DATA
    ACT_TARGET = TestClass.TARGET
    ACT_REFVECS = TestClass.REFVECS
    ACT_TRAIN = TestClass.TRAIN
    ACT_DEV = TestClass.DEV
    ACT_TEST = TestClass.TEST


    # TEST BY OBJECT
    NAMES = (f'DATA', f'TARGET', f'REFVECS')
    EXPECTEDS = (EXP_DATA, EXP_TARGET, EXP_REFVECS)
    ACTUALS = (ACT_DATA, ACT_TARGET, ACT_REFVECS)

    print(f'\033[92mTESTING FOR CONGRUITY OF GROUPINGS BY OBJECT...\033[0m')
    ctr = 0
    for name, actual, expected in zip(NAMES, ACTUALS, EXPECTEDS):
        for _type, _actual, _expected in zip(('TRAIN','DEV','TEST'),actual,expected):
            ctr += 1
            # print(f'Running trial {ctr} of {3*3}...')
            if not np.array_equiv(_actual, _expected):
                print(f'\033[91m')
                print(f'GIVEN {name}:')
                print({f'DATA':DATA, f'TARGET':TARGET, f'REFVECS':REFVECS}[name])
                print()
                print(f'EXPECTED {_type}:')
                print(_expected)
                print()
                print(f'ACTUAL {_type}')
                print(_actual)
                print()
                raise Exception(f'\n*** {name}, {_type} >>> INCONGRUITY BETWEEN ACTUAL AND EXPECTED ***\n')
                print(f'\033[0m')

    # TEST BY SPLIT
    NAMES = (f'TRAIN', f'DEV', f'TEST')
    EXPECTEDS = (EXP_TRAIN, EXP_DEV, EXP_TEST)
    ACTUALS = (ACT_TRAIN, ACT_DEV, ACT_TEST)

    print(f'\033[92mTESTING FOR CONGRUITY OF GROUPINGS BY SPLIT...\033[0m')
    ctr = 0
    for name, actual, expected in zip(NAMES, ACTUALS, EXPECTEDS):
        for _type, _actual, _expected in zip(('DATA','TARGET','REFVECS'),actual,expected):
            ctr += 1
            # print(f'Running trial {ctr} of {3*3}...')
            if not np.array_equiv(_actual, _expected):
                print(f'\033[91m')
                print(f'GIVEN {_type}:')
                print({f'DATA':DATA, f'TARGET':TARGET, f'REFVECS':REFVECS}[_type])
                print()
                print(f'EXPECTED {name}:')
                print(_expected)
                print()
                print(f'ACTUAL {name}')
                print(_actual)
                print()
                raise Exception(f'\n*** {name}, {_type} >>> INCONGRUITY BETWEEN ACTUAL AND EXPECTED ***\n')
                print(f'\033[0m')

    print(f'\033[92mmask() TESTS COMPLETED. ALL PASSED.\033[0m')
    # END TEST mask() ######################################################################################################
    ########################################################################################################################



    ########################################################################################################################
    # TEST random() ##########################################################################################################
    print(f'\033[92m\nSTARTING random() TESTS...\033[0m')

    # ESTABLISH THAT counts/percents WORK CORRECTLY AND MANUAL ENTRY WORKS
    # random(dev_count=None, dev_percent=None, test_count=None, test_percent=None)

    DATA = np.random.randint(0, 10, (3, 100))
    TARGET = np.random.randint(0, 10, (3, 100))
    REFVECS = np.random.randint(0, 10, (3, 100))
    _orient = 'COLUMN'

    TestClass = TrainDevTestSplit(DATA=DATA,
                                  TARGET=TARGET,
                                  REFVECS=REFVECS,
                                  data_given_orientation=_orient,
                                  target_given_orientation=_orient,
                                  refvecs_given_orientation=_orient,
                                  bypass_validation=False)

    # VERIFY percent GIVES CORRECT SIZES
    TestClass.random(dev_count=None, dev_percent=20, test_count=None, test_percent=10)
    exp_train_len = 70
    exp_dev_len = 20
    exp_test_len = 10

    for num, TRAIN_OBJ in enumerate((TestClass.DATA[0], TestClass.TARGET[0], TestClass.REFVECS[0], TestClass.TRAIN[0], TestClass.TRAIN[1], TestClass.TRAIN[2])):
        if not len(TRAIN_OBJ[0]) == exp_train_len:
            time.sleep(1); raise Exception(f'\033[91mTRAIN OBJECT NUMBER {num} len ({len(TRAIN_OBJ[0])}) DOES NOT MATCH EXPECTED ({exp_train_len})\033[0m')
    else: print(f'\033[92mrandom() percent TRAIN LENGTH PASSES\033[0m')

    for num, DEV_OBJ in enumerate((TestClass.DATA[1], TestClass.TARGET[1], TestClass.REFVECS[1], TestClass.DEV[0], TestClass.DEV[1], TestClass.DEV[2])):
        if not len(DEV_OBJ[0]) == exp_dev_len:
            time.sleep(1); raise Exception(f'\033[91mDEV OBJECT NUMBER {num} len ({len(DEV_OBJ[0])}) DOES NOT MATCH EXPECTED ({exp_dev_len})\033[0m')
    else: print(f'\033[92mrandom() percent DEV LENGTH PASSES\033[0m')

    for num, TEST_OBJ in enumerate((TestClass.DATA[2], TestClass.TARGET[2], TestClass.REFVECS[2], TestClass.TEST[0], TestClass.TEST[1], TestClass.TEST[2])):
        if not len(TEST_OBJ[0]) == exp_test_len:
            time.sleep(1); raise Exception(f'\033[91mTEST OBJECT NUMBER {num} len ({len(TEST_OBJ[0])}) DOES NOT MATCH EXPECTED ({exp_test_len})\033[0m')
    else: print(f'\033[92mrandom() percent TEST LENGTH PASSES\033[0m')

    # VERIFY count GIVES CORRECT SIZES
    TestClass.random(dev_count=25, dev_percent=None, test_count=15, test_percent=None)
    exp_train_len = 60
    exp_dev_len = 25
    exp_test_len = 15
    for num, TRAIN_OBJ in enumerate((TestClass.DATA[0], TestClass.TARGET[0], TestClass.REFVECS[0], TestClass.TRAIN[0], TestClass.TRAIN[1], TestClass.TRAIN[2])):
        if not len(TRAIN_OBJ[0]) == exp_train_len:
            time.sleep(1); raise Exception(f'\033[91mTRAIN OBJECT NUMBER {num} len ({len(TRAIN_OBJ[0])}) DOES NOT MATCH EXPECTED ({exp_train_len})\033[0m')
    else: print(f'\033[92mrandom() count TRAIN LENGTH PASSES\033[0m')

    for num, DEV_OBJ in enumerate((TestClass.DATA[1], TestClass.TARGET[1], TestClass.REFVECS[1], TestClass.DEV[0], TestClass.DEV[1], TestClass.DEV[2])):
        if not len(DEV_OBJ[0]) == exp_dev_len:
            time.sleep(1); raise Exception(f'\033[91mDEV OBJECT NUMBER {num} len ({len(DEV_OBJ[0])}) DOES NOT MATCH EXPECTED ({exp_dev_len})\033[0m')
    else: print(f'\033[92mrandom() count DEV LENGTH PASSES\033[0m')

    for num, TEST_OBJ in enumerate((TestClass.DATA[2], TestClass.TARGET[2], TestClass.REFVECS[2], TestClass.TEST[0], TestClass.TEST[1], TestClass.TEST[2])):
        if not len(TEST_OBJ[0]) == exp_test_len:
            time.sleep(1); raise Exception(f'\033[91mTEST OBJECT NUMBER {num} len ({len(TEST_OBJ[0])}) DOES NOT MATCH EXPECTED ({exp_test_len})\033[0m')
    else: print(f'\033[92mrandom() count TEST LENGTH PASSES\033[0m')

    # VERIFY MANUAL ENTRY WORKS
    TestClass.random()
    print(f'\033[92mrandom() MANUAL ENTRY TEST PASSES\033[0m')

    print(f'\033[92mrandom() TESTS COMPLETED. ALL PASSED.\033[0m')
    # END TEST random() ######################################################################################################
    ########################################################################################################################

    ########################################################################################################################
    # TEST partition() ##########################################################################################################
    print(f'\033[92m\nSTARTING partition() TESTS...\033[0m')

    # partition(self, number_of_partitions=None, dev_partition_number=None, test_partition_number=None)
    _rows = 21
    DATA = np.random.randint(0, 10, (3, _rows))
    TARGET = np.random.randint(0, 10, (3, _rows))
    REFVECS = np.random.randint(0, 10, (3, _rows))
    _orient = 'COLUMN'

    TestClass = TrainDevTestSplit(DATA=DATA,
                                  TARGET=TARGET,
                                  REFVECS=REFVECS,
                                  data_given_orientation=_orient,
                                  target_given_orientation=_orient,
                                  refvecs_given_orientation=_orient,
                                  bypass_validation=False)

    # TEST THAT DEV AND TEST PARTITIONS ARE CORRECT
    print(f'\033[92mpartition() TESTING FOR CONGRUITY FOR TRAIN/DEV/TEST PARTITIONING...\033[0m')
    num_partitions = 4
    part_size = _rows // num_partitions
    for dev_partition, test_partition in ((0,1),(1,2),(2,3),(3,0)):
        DEV_IDXS = range(part_size * dev_partition,part_size * (dev_partition + 1) if dev_partition != num_partitions - 1 else _rows)
        TEST_IDXS = range(part_size * test_partition,part_size * (test_partition + 1) if test_partition != num_partitions - 1 else _rows)
        TRAIN_MASK = np.fromiter((idx in [*DEV_IDXS, *TEST_IDXS] for idx in range(_rows)), dtype=bool)
        DEV_MASK = np.fromiter((idx in DEV_IDXS for idx in range(_rows)), dtype=bool)
        TEST_MASK = np.fromiter((idx in TEST_IDXS for idx in range(_rows)), dtype=bool)

        EXP_DATA_TRAIN = np.delete(DATA, TRAIN_MASK, axis=1)
        EXP_DATA_DEV = DATA[..., DEV_MASK]
        EXP_DATA_TEST = DATA[..., TEST_MASK]
        EXP_TARGET_TRAIN = np.delete(TARGET, TRAIN_MASK, axis=1)
        EXP_TARGET_DEV = TARGET[..., DEV_MASK]
        EXP_TARGET_TEST = TARGET[..., TEST_MASK]
        EXP_REFVECS_TRAIN = np.delete(REFVECS, TRAIN_MASK, axis=1)
        EXP_REFVECS_DEV = REFVECS[..., DEV_MASK]
        EXP_REFVECS_TEST = REFVECS[..., TEST_MASK]

        TestClass.partition(number_of_partitions=num_partitions,
                            dev_partition_number=dev_partition,
                            test_partition_number=test_partition)

        ACT_DATA_TRAIN = TestClass.TRAIN[0]
        ACT_TARGET_TRAIN = TestClass.TRAIN[1]
        ACT_REFVECS_TRAIN = TestClass.TRAIN[2]
        ACT_DATA_DEV = TestClass.DEV[0]
        ACT_TARGET_DEV = TestClass.DEV[1]
        ACT_REFVECS_DEV = TestClass.DEV[2]
        ACT_DATA_TEST = TestClass.TEST[0]
        ACT_TARGET_TEST = TestClass.TEST[1]
        ACT_REFVECS_TEST = TestClass.TEST[2]

        NAMES = ('DATA_TRAIN', 'TARGET_TRAIN', 'REFVECS_TRAIN', 'DATA_DEV', 'TARGET_DEV', 'REFVECS_DEV', 'DATA_TEST', 'TARGET_TEST', 'REFVECS_TEST')
        EXP_OBJS = (EXP_DATA_TRAIN, EXP_TARGET_TRAIN, EXP_REFVECS_TRAIN, EXP_DATA_DEV, EXP_TARGET_DEV, EXP_REFVECS_DEV, EXP_DATA_TEST,
                    EXP_TARGET_TEST, EXP_REFVECS_TEST)
        ACT_OBJS = (ACT_DATA_TRAIN, ACT_TARGET_TRAIN, ACT_REFVECS_TRAIN, ACT_DATA_DEV, ACT_TARGET_DEV, ACT_REFVECS_DEV, ACT_DATA_TEST,
                    ACT_TARGET_TEST, ACT_REFVECS_TEST)

        for name, exp_obj, act_obj in zip(NAMES, EXP_OBJS, ACT_OBJS):
            if not np.array_equiv(exp_obj, act_obj):
                print(f'\033[91m')
                # print(f'GIVEN {type}:')
                print(f'DATA:')
                print(DATA)
                print()
                print(f'TARGET:')
                print(TARGET)
                print()
                print(f'REFVECS')
                print(REFVECS)
                print()
                print(f'EXPECTED {name}:')
                print(exp_obj)
                print()
                print(f'ACTUAL {name}')
                print(act_obj)
                print()
                raise Exception(f'\n*** partition() >>> {name}  INCONGRUITY BETWEEN ACTUAL AND EXPECTED ***\n')
                print(f'\033[0m')

    # TEST THAT TEST PARTITIONS ARE CORRECT
    print(f'\033[92mpartition() TESTING FOR CONGRUITY FOR TRAIN/TEST PARTITIONING VIA DEV KWARGS...\033[0m')
    num_partitions = 4
    part_size = _rows // num_partitions
    for dev_partition in (0,1,2,3):
        DEV_IDXS = range(part_size * dev_partition,part_size * (dev_partition + 1) if dev_partition != num_partitions - 1 else _rows)
        DEV_MASK = np.fromiter((idx in DEV_IDXS for idx in range(_rows)), dtype=bool)

        EXP_DATA_TRAIN = np.delete(DATA, DEV_MASK, axis=1)
        EXP_DATA_TEST = DATA[..., DEV_MASK]
        EXP_TARGET_TRAIN = np.delete(TARGET, DEV_MASK, axis=1)
        EXP_TARGET_TEST = TARGET[..., DEV_MASK]
        EXP_REFVECS_TRAIN = np.delete(REFVECS, DEV_MASK, axis=1)
        EXP_REFVECS_TEST = REFVECS[..., DEV_MASK]

        TestClass.partition(number_of_partitions=num_partitions,
                            dev_partition_number=dev_partition,
                            test_partition_number=None)

        ACT_DATA_TRAIN = TestClass.TRAIN[0]
        ACT_TARGET_TRAIN = TestClass.TRAIN[1]
        ACT_REFVECS_TRAIN = TestClass.TRAIN[2]
        ACT_DATA_TEST = TestClass.TEST[0]
        ACT_TARGET_TEST = TestClass.TEST[1]
        ACT_REFVECS_TEST = TestClass.TEST[2]

        NAMES = ('DATA_TRAIN', 'TARGET_TRAIN', 'REFVECS_TRAIN', 'DATA_TEST', 'TARGET_TEST', 'REFVECS_TEST')
        EXP_OBJS = (EXP_DATA_TRAIN, EXP_TARGET_TRAIN, EXP_REFVECS_TRAIN, EXP_DATA_TEST, EXP_TARGET_TEST, EXP_REFVECS_TEST)
        ACT_OBJS = (ACT_DATA_TRAIN, ACT_TARGET_TRAIN, ACT_REFVECS_TRAIN, ACT_DATA_TEST, ACT_TARGET_TEST, ACT_REFVECS_TEST)

        for name, exp_obj, act_obj in zip(NAMES, EXP_OBJS, ACT_OBJS):
            if not np.array_equiv(exp_obj, act_obj):
                print(f'\033[91m')
                # print(f'GIVEN {type}:')
                print(f'DATA:')
                print(DATA)
                print()
                print(f'TARGET:')
                print(TARGET)
                print()
                print(f'REFVECS')
                print(REFVECS)
                print()
                print(f'EXPECTED {name}:')
                print(exp_obj)
                print()
                print(f'ACTUAL {name}')
                print(act_obj)
                print()
                raise Exception(f'\n*** partition() >>> {name}  INCONGRUITY BETWEEN ACTUAL AND EXPECTED ***\n')
                print(f'\033[0m')


    print(f'\033[92mpartition() TESTING FOR CONGRUITY FOR TRAIN/TEST PARTITIONING VIA TEST KWARGS...\033[0m')
    # NOTE TESTING ALSO TESTING ONLY 2 GIVEN OBJECTS INSTEAD OF 3

    TestClass = TrainDevTestSplit(DATA=DATA,
                                  TARGET=None,
                                  REFVECS=REFVECS,
                                  data_given_orientation=_orient,
                                  target_given_orientation=None,
                                  refvecs_given_orientation=_orient,
                                  bypass_validation=False)

    num_partitions = 4
    part_size = _rows // num_partitions
    for test_partition in (0,1,2,3):
        TEST_IDXS = range(part_size * test_partition,part_size * (test_partition + 1) if test_partition != num_partitions - 1 else _rows)
        TEST_MASK = np.fromiter((idx in TEST_IDXS for idx in range(_rows)), dtype=bool)

        EXP_DATA_TRAIN = np.delete(DATA, TEST_MASK, axis=1)
        EXP_DATA_TEST = DATA[..., TEST_MASK]
        # EXP_TARGET_TRAIN = np.delete(TARGET, TEST_MASK, axis=1)
        # EXP_TARGET_TEST = TARGET[..., TEST_MASK]
        EXP_REFVECS_TRAIN = np.delete(REFVECS, TEST_MASK, axis=1)
        EXP_REFVECS_TEST = REFVECS[..., TEST_MASK]

        TestClass.partition(number_of_partitions=num_partitions,
                            dev_partition_number=None,
                            test_partition_number=test_partition)

        ACT_DATA_TRAIN = TestClass.DATA[0]
        ACT_DATA_TEST = TestClass.DATA[1]
        ACT_REFVECS_TRAIN = TestClass.REFVECS[0]
        ACT_REFVECS_TEST = TestClass.REFVECS[1]

        NAMES = ('DATA_TRAIN', 'REFVECS_TRAIN', 'DATA_TEST', 'REFVECS_TEST')
        EXP_OBJS = (EXP_DATA_TRAIN, EXP_REFVECS_TRAIN, EXP_DATA_TEST, EXP_REFVECS_TEST)
        ACT_OBJS = (ACT_DATA_TRAIN, ACT_REFVECS_TRAIN, ACT_DATA_TEST, ACT_REFVECS_TEST)

        for name, exp_obj, act_obj in zip(NAMES, EXP_OBJS, ACT_OBJS):
            if not np.array_equiv(exp_obj, act_obj):
                print(f'\033[91m')
                # print(f'GIVEN {type}:')
                print(f'DATA:')
                print(DATA)
                print()
                print(f'REFVECS')
                print(REFVECS)
                print()
                print(f'EXPECTED {name}:')
                print(exp_obj)
                print()
                print(f'ACTUAL {name}')
                print(act_obj)
                print()
                raise Exception(f'\n*** partition() >>> {name}  INCONGRUITY BETWEEN ACTUAL AND EXPECTED ***\n')
                print(f'\033[0m')

    print(f'\033[92mpartition() MANUAL ENTRY TEST...\033[0m')
    TestClass.partition()

    print(f'\033[92mpartition() TESTS COMPLETED. ALL PASSED.\033[0m')
    # END TEST partition() ######################################################################################################
    ########################################################################################################################

    ########################################################################################################################
    # TEST category() ##########################################################################################################
    print(f'\033[92m\nSTARTING category() TESTS...\033[0m')

    from MLObjects.SupportObjects import BuildFullSupportObject as bfso

    _rows = 100
    DATA = np.random.randint(0, 10, (3, _rows))
    TARGET = np.random.randint(0, 10, (3, _rows))
    REFVECS = np.random.randint(0, 10, (3, _rows))
    _orient = 'COLUMN'

    # PUT FILTER COLUMN ON DATA
    DATA = np.vstack((DATA, np.random.choice(['TRAIN','DEV','TEST'], _rows, replace=True, p=[0.6,0.3,0.1])))

    KWARGS = dict((('object_given_orientation',_orient), ('OBJECT_HEADER',None), ('SUPPORT_OBJECT',None), ('columns',None),
                  ('quick_vdtypes',False), ('MODIFIED_DATATYPES',None), ('print_notes',False), ('prompt_to_override',False),
                  ('bypass_validation',False), ('calling_module','TrainDevTestSplit'), ('calling_fxn','category()_test')))

    # GET SUPOBJS
    DATA_SUPOBJ = bfso.BuildFullSupportObject(OBJECT=DATA, **KWARGS).SUPPORT_OBJECT
    TARGET_SUPOBJ = bfso.BuildFullSupportObject(OBJECT=TARGET, **KWARGS).SUPPORT_OBJECT
    REFVECS_SUPOBJ = bfso.BuildFullSupportObject(OBJECT=REFVECS, **KWARGS).SUPPORT_OBJECT

    del KWARGS

    TRAIN_MASK = DATA[-1] == 'TRAIN'
    DEV_MASK = DATA[-1] == 'DEV'
    TEST_MASK = DATA[-1] == 'TEST'

    EXP_DATA_TRAIN = DATA[..., TRAIN_MASK]
    EXP_TARGET_TRAIN = TARGET[..., TRAIN_MASK]
    EXP_REFVECS_TRAIN = REFVECS[..., TRAIN_MASK]
    EXP_DATA_DEV = DATA[..., DEV_MASK]
    EXP_TARGET_DEV = TARGET[..., DEV_MASK]
    EXP_REFVECS_DEV = REFVECS[..., DEV_MASK]
    EXP_DATA_TEST = DATA[..., TEST_MASK]
    EXP_TARGET_TEST = TARGET[..., TEST_MASK]
    EXP_REFVECS_TEST = REFVECS[..., TEST_MASK]


    print(f'\033[92mcategory() TEST WITH GIVEN COLUMN/VALUE AND WITHOUT SUPOBJS...\033[0m')
    TestClass = TrainDevTestSplit(DATA=DATA,
                                  TARGET=TARGET,
                                  REFVECS=REFVECS,
                                  data_given_orientation=_orient,
                                  target_given_orientation=_orient,
                                  refvecs_given_orientation=_orient,
                                  bypass_validation=False)

    TestClass.category(object_name_for_dev='DATA',
                       dev_column_name_or_idx=-1,
                       DEV_SPLIT_ON_VALUES_AS_LIST=['DEV'],
                       object_name_for_test='DATA',
                       test_column_name_or_idx=-1,
                       TEST_SPLIT_ON_VALUES_AS_LIST=['TEST'],
                       DATA_FULL_SUPOBJ=None,
                       TARGET_FULL_SUPOBJ=None,
                       REFVECS_FULL_SUPOBJ=None)

    ACT_DATA_TRAIN = TestClass.TRAIN[0]
    ACT_TARGET_TRAIN = TestClass.TRAIN[1]
    ACT_REFVECS_TRAIN = TestClass.TRAIN[2]
    ACT_DATA_DEV = TestClass.DEV[0]
    ACT_TARGET_DEV = TestClass.DEV[1]
    ACT_REFVECS_DEV = TestClass.DEV[2]
    ACT_DATA_TEST = TestClass.TEST[0]
    ACT_TARGET_TEST = TestClass.TEST[1]
    ACT_REFVECS_TEST = TestClass.TEST[2]


    NAMES = ('DATA_TRAIN', 'DATA_DEV', 'DATA_TEST', 'TARGET_TRAIN ','TARGET_DEV','TARGET_TEST','REFVECS_TRAIN', 'REFVECS_DEV', 'REFVECS_TEST')
    EXPECTEDS = (EXP_DATA_TRAIN, EXP_TARGET_TRAIN, EXP_REFVECS_TRAIN, EXP_DATA_DEV, EXP_TARGET_DEV, EXP_REFVECS_DEV,
                 EXP_DATA_TEST, EXP_TARGET_TEST, EXP_REFVECS_TEST)
    ACTUALS = (ACT_DATA_TRAIN, ACT_TARGET_TRAIN, ACT_REFVECS_TRAIN, ACT_DATA_DEV, ACT_TARGET_DEV, ACT_REFVECS_DEV,
               ACT_DATA_TEST, ACT_TARGET_TEST, ACT_REFVECS_TEST)

    for name, exp_obj, act_obj in zip(NAMES, EXPECTEDS, ACTUALS):
        if not np.array_equiv(exp_obj, act_obj):
            print(f'\033[91m')
            print(f'GIVEN DATA:')
            print(DATA)
            print(f'GIVEN TARGET:')
            print(TARGET)
            print(f'GIVEN REFVECS:')
            print(REFVECS)
            print(f'EXPECTED {name}:')
            print(exp_obj)
            print(f'ACTUAL {name}:')
            print(act_obj)
            raise Exception(f'INCONGRUITY BEWTEEN {name} EXPECTED AND ACTUAL')
            print(f'\033[0m')



    print(f'\033[92mcategory() TEST WITH GIVEN COLUMN/VALUE AND WITH SUPOBJS...\033[0m')

    TestClass = TrainDevTestSplit(DATA=DATA,
                                  TARGET=TARGET,
                                  REFVECS=REFVECS,
                                  data_given_orientation=_orient,
                                  target_given_orientation=_orient,
                                  refvecs_given_orientation=_orient,
                                  bypass_validation=False)

    TestClass.category(object_name_for_dev='DATA',
                       dev_column_name_or_idx=-1,
                       DEV_SPLIT_ON_VALUES_AS_LIST=['DEV'],
                       object_name_for_test='DATA',
                       test_column_name_or_idx=-1,
                       TEST_SPLIT_ON_VALUES_AS_LIST=['TEST'],
                       DATA_FULL_SUPOBJ=DATA_SUPOBJ,
                       TARGET_FULL_SUPOBJ=TARGET_SUPOBJ,
                       REFVECS_FULL_SUPOBJ=REFVECS_SUPOBJ)

    ACT_DATA_TRAIN = TestClass.TRAIN[0]
    ACT_TARGET_TRAIN = TestClass.TRAIN[1]
    ACT_REFVECS_TRAIN = TestClass.TRAIN[2]
    ACT_DATA_DEV = TestClass.DEV[0]
    ACT_TARGET_DEV = TestClass.DEV[1]
    ACT_REFVECS_DEV = TestClass.DEV[2]
    ACT_DATA_TEST = TestClass.TEST[0]
    ACT_TARGET_TEST = TestClass.TEST[1]
    ACT_REFVECS_TEST = TestClass.TEST[2]

    NAMES = ('DATA_TRAIN', 'DATA_DEV', 'DATA_TEST', 'TARGET_TRAIN ','TARGET_DEV','TARGET_TEST','REFVECS_TRAIN', 'REFVECS_DEV', 'REFVECS_TEST')
    EXPECTEDS = (EXP_DATA_TRAIN, EXP_TARGET_TRAIN, EXP_REFVECS_TRAIN, EXP_DATA_DEV, EXP_TARGET_DEV, EXP_REFVECS_DEV,
                 EXP_DATA_TEST, EXP_TARGET_TEST, EXP_REFVECS_TEST)
    ACTUALS = (ACT_DATA_TRAIN, ACT_TARGET_TRAIN, ACT_REFVECS_TRAIN, ACT_DATA_DEV, ACT_TARGET_DEV, ACT_REFVECS_DEV,
               ACT_DATA_TEST, ACT_TARGET_TEST, ACT_REFVECS_TEST)

    for name, exp_obj, act_obj in zip(NAMES, EXPECTEDS, ACTUALS):
        if not np.array_equiv(exp_obj, act_obj):
            print(f'\033[91m')
            print(f'GIVEN DATA:')
            print(DATA)
            print(f'GIVEN TARGET:')
            print(TARGET)
            print(f'GIVEN REFVECS:')
            print(REFVECS)
            print(f'EXPECTED {name}:')
            print(exp_obj)
            print(f'ACTUAL {name}:')
            print(act_obj)
            raise Exception(f'INCONGRUITY BEWTEEN {name} EXPECTED AND ACTUAL')
            print(f'\033[0m')



    print(f'\033[92mcategory() TEST WITH ONLY SUBOBJS GIVEN...\033[0m')
    TestClass = TrainDevTestSplit(DATA=DATA,
                                  TARGET=TARGET,
                                  REFVECS=REFVECS,
                                  data_given_orientation=_orient,
                                  target_given_orientation=_orient,
                                  refvecs_given_orientation=_orient,
                                  bypass_validation=False)

    TestClass.category(object_name_for_dev=None,
                       dev_column_name_or_idx=None,
                       DEV_SPLIT_ON_VALUES_AS_LIST=None,
                       object_name_for_test=None,
                       test_column_name_or_idx=None,
                       TEST_SPLIT_ON_VALUES_AS_LIST=None,
                       DATA_FULL_SUPOBJ=DATA_SUPOBJ,
                       TARGET_FULL_SUPOBJ=TARGET_SUPOBJ,
                       REFVECS_FULL_SUPOBJ=REFVECS_SUPOBJ)

    ACT_DATA_TRAIN = TestClass.TRAIN[0]
    ACT_TARGET_TRAIN = TestClass.TRAIN[1]
    ACT_REFVECS_TRAIN = TestClass.TRAIN[2]
    ACT_DATA_DEV = TestClass.DEV[0]
    ACT_TARGET_DEV = TestClass.DEV[1]
    ACT_REFVECS_DEV = TestClass.DEV[2]
    ACT_DATA_TEST = TestClass.TEST[0]
    ACT_TARGET_TEST = TestClass.TEST[1]
    ACT_REFVECS_TEST = TestClass.TEST[2]

    NAMES = ('DATA_TRAIN', 'DATA_DEV', 'DATA_TEST', 'TARGET_TRAIN ','TARGET_DEV','TARGET_TEST','REFVECS_TRAIN', 'REFVECS_DEV', 'REFVECS_TEST')
    EXPECTEDS = (EXP_DATA_TRAIN, EXP_TARGET_TRAIN, EXP_REFVECS_TRAIN, EXP_DATA_DEV, EXP_TARGET_DEV, EXP_REFVECS_DEV,
                 EXP_DATA_TEST, EXP_TARGET_TEST, EXP_REFVECS_TEST)
    ACTUALS = (ACT_DATA_TRAIN, ACT_TARGET_TRAIN, ACT_REFVECS_TRAIN, ACT_DATA_DEV, ACT_TARGET_DEV, ACT_REFVECS_DEV,
               ACT_DATA_TEST, ACT_TARGET_TEST, ACT_REFVECS_TEST)

    for name, exp_obj, act_obj in zip(NAMES, EXPECTEDS, ACTUALS):
        if not np.array_equiv(exp_obj, act_obj):
            print(f'\033[91m')
            print(f'GIVEN DATA:')
            print(DATA)
            print(f'GIVEN TARGET:')
            print(TARGET)
            print(f'GIVEN REFVECS:')
            print(REFVECS)
            print(f'EXPECTED {name}:')
            print(exp_obj)
            print(f'ACTUAL {name}:')
            print(act_obj)
            raise Exception(f'INCONGRUITY BEWTEEN {name} EXPECTED AND ACTUAL')
            print(f'\033[0m')


    print(f'\033[92mcategory() TEST WITH NOTHING PASSED TO category() KWARGS...\033[0m')
    TestClass = TrainDevTestSplit(DATA=DATA,
                                  TARGET=TARGET,
                                  REFVECS=REFVECS,
                                  data_given_orientation=_orient,
                                  target_given_orientation=_orient,
                                  refvecs_given_orientation=_orient,
                                  bypass_validation=False)

    TestClass.category(object_name_for_dev=None,
                       dev_column_name_or_idx=None,
                       DEV_SPLIT_ON_VALUES_AS_LIST=None,
                       object_name_for_test=None,
                       test_column_name_or_idx=None,
                       TEST_SPLIT_ON_VALUES_AS_LIST=None,
                       DATA_FULL_SUPOBJ=None,
                       TARGET_FULL_SUPOBJ=None,
                       REFVECS_FULL_SUPOBJ=None)

    ACT_DATA_TRAIN = TestClass.TRAIN[0]
    ACT_TARGET_TRAIN = TestClass.TRAIN[1]
    ACT_REFVECS_TRAIN = TestClass.TRAIN[2]
    ACT_DATA_DEV = TestClass.DEV[0]
    ACT_TARGET_DEV = TestClass.DEV[1]
    ACT_REFVECS_DEV = TestClass.DEV[2]
    ACT_DATA_TEST = TestClass.TEST[0]
    ACT_TARGET_TEST = TestClass.TEST[1]
    ACT_REFVECS_TEST = TestClass.TEST[2]

    NAMES = ('DATA_TRAIN', 'DATA_DEV', 'DATA_TEST', 'TARGET_TRAIN ','TARGET_DEV','TARGET_TEST','REFVECS_TRAIN', 'REFVECS_DEV', 'REFVECS_TEST')
    EXPECTEDS = (EXP_DATA_TRAIN, EXP_TARGET_TRAIN, EXP_REFVECS_TRAIN, EXP_DATA_DEV, EXP_TARGET_DEV, EXP_REFVECS_DEV,
                 EXP_DATA_TEST, EXP_TARGET_TEST, EXP_REFVECS_TEST)
    ACTUALS = (ACT_DATA_TRAIN, ACT_TARGET_TRAIN, ACT_REFVECS_TRAIN, ACT_DATA_DEV, ACT_TARGET_DEV, ACT_REFVECS_DEV,
               ACT_DATA_TEST, ACT_TARGET_TEST, ACT_REFVECS_TEST)

    for name, exp_obj, act_obj in zip(NAMES, EXPECTEDS, ACTUALS):
        if not np.array_equiv(exp_obj, act_obj):
            print(f'\033[91m')
            print(f'GIVEN DATA:')
            print(DATA)
            print(f'GIVEN TARGET:')
            print(TARGET)
            print(f'GIVEN REFVECS:')
            print(REFVECS)
            print(f'EXPECTED {name}:')
            print(exp_obj)
            print(f'ACTUAL {name}:')
            print(act_obj)
            raise Exception(f'INCONGRUITY BEWTEEN {name} EXPECTED AND ACTUAL')
            print(f'\033[0m')



    print(f'\033[92mcategory() TESTS COMPLETED. ALL PASSED.\033[0m')
    # END TEST category() ######################################################################################################
    ########################################################################################################################


    print(f'\033[92m\nTESTS COMPLETE. ALL PASSED.\033[0m')
    for _ in range(3): wls.winlinsound(888,500); time.sleep(1)























