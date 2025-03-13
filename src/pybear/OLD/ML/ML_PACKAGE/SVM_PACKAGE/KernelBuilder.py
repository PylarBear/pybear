import numpy as n, pandas as p, sys, warnings, psutil, os
from debug import get_module_name as gmn; from data_validation import validate_user_input as vui
import MemSizes as ms
import sparse_dict as sd
from ML_PACKAGE.SVM_PACKAGE import svm_kernels as sk


class KernelBuilder:
    def __init__(self, DATA, kernel_fxn='LINEAR', constant=0, exponent=1, sigma=1, DATA_HEADER=None, prompt_bypass=False,
                 run_data_as=None, return_kernel_as=None):

        # IF prompt_bypass IS False, MEANS NOT RUNNING K-FOLD, SO DONT NEED TO GIVE run_data_as AND return_kernel_as (USER WILL BE PROMPTED)
        # IF True, run_data_as AND return_kernel_as ARE HANDLED BASED ON WHETHER THEY WERE GIVEN AS KWARGS OR NOT

        self.module = gmn.get_module_name(str(sys.modules[__name__]))

        # DATA MUST COME IN AS [ [] = ROWS ], OR AS SPARSE DICT WITH { {} = ROWS }

        self.DATA = DATA
        self.data_dtype = ''

        self.kernel_fxn = kernel_fxn.upper()
        if self.kernel_fxn not in ['LINEAR', 'POLYNOMIAL', 'GAUSSIAN']:
            raise ValueError(f'\nINVALID kernel_fxn IN KernelBuilder, {self.kernel_fxn}.  Must be "LINEAR", "POLYNOMIAL", or "GAUSSIAN".')
        self.constant = constant
        self.exponent = exponent
        self.sigma = sigma
        self.DATA_HEADER = DATA_HEADER
        self.prompt_bypass = prompt_bypass

        if not run_data_as is None:
            self.run_data_as = run_data_as.upper()
            if self.run_data_as not in ['ARRAY', 'SPARSE_DICT']:
                raise TypeError(f'\n *** KernelBuilder() run_data_as IS "{self.run_data_as}".  MUST BE "ARRAY" OR "SPARSE_DICT". ***\n')
        else:
            self.run_data_as = run_data_as

        if not return_kernel_as is None:
            self.return_kernel_as = return_kernel_as.upper()
            if self.return_kernel_as not in ['ARRAY', 'SPARSE_DICT']:
                raise TypeError(f'\n *** KernelBuilder() return_kernel_as IS "{self.return_kernel_as}".  MUST BE "ARRAY" OR "SPARSE_DICT". ***\n')
        else:
            self.return_kernel_as = return_kernel_as

        self.is_list = isinstance(self.DATA, (list,tuple,n.ndarray))
        self.is_dict = isinstance(self.DATA, dict)

        # PLACEHOLDERS
        self.start_sparsity = 0
        self._outer_len = 0
        self.data_size = 0
        self.number_of_data_elements = 0
        # FOR PREDICTING K SPARSITY FROM DATA ATTRIBUTES ####
        self.data_ct_cat_columns = 0
        self.data_min_cat_sparsity = 0
        self.data_max_cat_sparsity = 0
        self.data_min_float_sparsity = 0
        self.data_max_float_sparsity = 0
        '''
        self.data_avg_cat_sparsity = 0
        self.data_ct_float_columns = 0
        self.data_avg_float_sparsity = 0
        '''

        ######################################################
        self.data_mem_size_as_array = 0
        self.data_mem_size_as_sd = 0
        self.K_size = 0
        self.K_pred_mem_size_as_sd = 0
        self.full_K_pred_mem_size_as_sd = 0
        self.K_pred_elements = 0
        self.K_pred_sparsity = 0
        self.K_act_mem_size_as_array = 0
        self.K_act_mem_size_as_sd = 0
        self.K_act_elements = 0
        self.K_act_sparsity = 0

        self.return_as = ''

        # CUTOFFS
        self.run_data_as_sparse_dict = False
        self.sparsity_cutoff = 85
        self.memory_cutoff = 3072 #MB
        self.array_size_cutoff = 400e6
        self.sd_size_cutoff = 30e6

        # BUILD K FILLED W ZEROS FOR FILLING LATER
        self.K = n.zeros((len(self.DATA), len(self.DATA)), dtype=object)

        # TESTING SHOWS FOR MATMUL W SYMMETRIC RESULT, SPARSE DICTS ARE FASTER THAN n.matmul WHEN SPARSITY IS ~ >60%.
        # SOMETHING SEEMS TO HAVE CHANGED :( AND THIS DOESN'T SEEM TO BE TRUE ANYMORE

        # K IS ALWAYS SYMMETRIC.  SEE sparse_dict

        while True:

            # CONFIGURE DATA BEFORE BUILDING KERNEL, THIS DOES NOT DETERMINE IF K IS array or sd! ##############################
            # ONLY WHETHER TO TRANSFORM DATA BEFORE BUILDING K!
            try:
                sd.sparse_dict_check(self.DATA, '')  # IF A SPARSE_DICT
                self.data_dtype = 'SPARSE DICT'
                break
            except: pass

            try:
                sd.list_check(self.DATA, '')  # IF A LIST-TYPE
                self.data_dtype = 'LIST-TYPE'
                break
            except: pass

            try:
                sd.dataframe_check(self.DATA, '')  # IF A DATAFRAME. CONVERT TO NUMPY FIRST.
                self.data_dtype = 'DATAFRAME'
                self.DATA = self.DATA.to_numpy(dtype=float)
                break
            except: pass

            # IF MAKE IT TO THIS POINT, DATA WAS NOT LIST-TYPE, SPARSE DICT OR DF.
            raise TypeError(f'SVM_PACKAGE.KernelBuilder() INCOMING DATA IS INVALID, OR IMPROPERLY FORMATTED. '
                                      f'MUST BE LIST-TYPE, SPARSE DICT, OR DF')

        self.calculate_properties_of_data_object()
        self.determine_best_data_format()

        # END DATA PREP FOR KERNEL BUILD ##############################################################################

    def calculate_properties_of_data_object(self):
        '''Calculate sparsity, length x width, and sparsity for each feature for DATA.'''
        if self.is_dict:
            self._outer_len = sd.outer_len(self.DATA)
            _inner_len = sd.inner_len(self.DATA)
            self.start_sparsity = round(sd.sparsity(self.DATA), 2)
            self.data_size = self._outer_len * _inner_len
            self.get_data_attributes(self.DATA)
        elif self.is_list:
            self._outer_len = len(self.DATA)
            _inner_len = len(self.DATA[0])
            self.start_sparsity = round(sd.list_sparsity(self.DATA), 2)
            self.data_size = self._outer_len * _inner_len
            self.get_data_attributes(self.DATA.transpose())

        self.number_of_data_elements = round(self.data_size * (100 - self.start_sparsity) / 100, 0)
        self.data_mem_size_as_array = ms.MemSizes('np_float').mb() * self.data_size
        self.data_mem_size_as_sd = ms.MemSizes('sd_float').mb() * self.number_of_data_elements
        print(f'\nDATA IS A {self._outer_len} x {_inner_len} {self.data_dtype} THAT IS {self.start_sparsity}% SPARSE.')


    # ONLY CALLED BY calculate_properties_of_data_object()
    def get_data_attributes(self, DATA_OBJECT):
        '''Calculate sparsity for each feature in DATA.'''

        if self.is_list:  # HAS BEEN TRANSPOSED TO [] = COLUMNS!
            CAT_SPARSITIES = n.zeros((1,0))[0]
            FLOAT_SPARSITIES = n.zeros((1,0))[0]
            for idx in range(len(DATA_OBJECT)):
                _sparsity = round(100 - 100 * len(n.nonzero(DATA_OBJECT[idx])[-1]) / len(DATA_OBJECT[idx]), 2)
                if n.min(DATA_OBJECT[idx])==0 and n.max(DATA_OBJECT[idx])==1:
                    CAT_SPARSITIES = n.insert(CAT_SPARSITIES, len(CAT_SPARSITIES), _sparsity)
                else:  # IF MIN != 0 AND MAX != 1, THEN IS A FLOAT COLUMN
                    FLOAT_SPARSITIES = n.insert(FLOAT_SPARSITIES, len(FLOAT_SPARSITIES), _sparsity)

        elif self.is_dict:  # HAS NOT BEEN TRANSPOSED!
            _outer_len, _inner_len = sd.outer_len(DATA_OBJECT), sd.inner_len(DATA_OBJECT)
            CAT_SPARSITIES = n.zeros((1, 0))[0]
            FLOAT_SPARSITIES = n.zeros((1, 0))[0]
            # SLICE THRU ALL OUTER IDXs BY EACH INNER IDX
            for inner_idx in range(_inner_len):   # INNER IDX = COLUMNS OF DATA!  DATA CAME IN AS [ [] = ROWS ]
                hits, _min, _max = 0, 0, 0
                for outer_idx in range(_outer_len):
                    if inner_idx != _inner_len-1 and inner_idx in DATA_OBJECT[outer_idx]:
                        hits += 1
                        _min = min(_min, DATA_OBJECT[outer_idx][inner_idx])
                        _max = max(_max, DATA_OBJECT[outer_idx][inner_idx])
                    elif inner_idx == _inner_len-1 and DATA_OBJECT[outer_idx][_inner_len - 1] != 0:
                        hits += 1
                        _min = min(_min, DATA_OBJECT[outer_idx][inner_idx])
                        _max = max(_max, DATA_OBJECT[outer_idx][inner_idx])

                _sparsity = round(100 - 100 * hits / _outer_len, 2)

                if _min==0 and _max==1:
                    CAT_SPARSITIES = n.insert(CAT_SPARSITIES, len(CAT_SPARSITIES), _sparsity)

                else:   # IF MIN != 0 AND MAX != 1, THEN IS A FLOAT COLUMN
                    FLOAT_SPARSITIES = n.insert(FLOAT_SPARSITIES, len(FLOAT_SPARSITIES), _sparsity)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if len(CAT_SPARSITIES) > 0:
                self.data_ct_cat_columns = len(CAT_SPARSITIES)
                self.data_min_cat_sparsity = n.min(CAT_SPARSITIES)
                self.data_max_cat_sparsity = n.max(CAT_SPARSITIES)
                self.data_avg_cat_sparsity = n.average(CAT_SPARSITIES)
            else:
                self.data_ct_cat_columns = 0
                self.data_min_cat_sparsity = 100
                self.data_max_cat_sparsity = 100
                self.data_avg_cat_sparsity = 100
            if len(FLOAT_SPARSITIES) > 0:
                self.data_ct_float_columns = len(FLOAT_SPARSITIES)
                self.data_min_float_sparsity = n.min(FLOAT_SPARSITIES)
                self.data_avg_float_sparsity = n.average(FLOAT_SPARSITIES)
            else:
                self.data_ct_float_columns = 0
                self.data_min_float_sparsity = 100
                self.data_avg_float_sparsity = 100

        del CAT_SPARSITIES, FLOAT_SPARSITIES


    def determine_best_data_format(self):
        '''Calculate best data format (array or sd) based on final size and speed considerations.'''
        if self.prompt_bypass is False:
            # BOTH EXCEED MEMORY ALLOWANCE --- ALLOW USER TO SELECT WHICH TYPE TO USE
            if self.data_mem_size_as_array > self.memory_cutoff and self.data_mem_size_as_sd > self.memory_cutoff:
                print(f'\nDATA AS ARRAY ({self.data_size:,.0f} elements, {self.start_sparsity}% sparsity, est {self.data_mem_size_as_array:,.1f} MB) '
                      f'AND AS SPARSE DICT ({self.number_of_data_elements:,.0f} elements, est {self.data_mem_size_as_sd:,.1f} MB) ' 
                      f'BOTH EXCEED MEMORY SIZE CUTOFF ({self.memory_cutoff:,.0f} MB). PICK YOUR POISON.')
                self.run_data_as_sparse_dict = vui.validate_user_str(f'Run as ARRAY(a) or SPARSE DICT(s) > ', 'AS') == 'S'
            # ARRAY LOWER THAN, SD MORE THAN MEMORY ALLOWANCE --- FORCE TO USE ARRAY
            elif self.data_mem_size_as_array <= self.memory_cutoff and self.data_mem_size_as_sd > self.memory_cutoff:
                print(f'\nDATA AS ARRAY ({self.data_size:,.0f} elements, est {self.data_mem_size_as_array:,.1f} MB) REQUIRES LESS MEMORY '
                  f'THAN SPARSE DICT ({self.number_of_data_elements:,.0f} elements, est {self.data_mem_size_as_sd:,.1f} MB). RUNNING AS ARRAY.')
                self.run_data_as_sparse_dict = False
            # ARRAY MORE THAN, SD LESS THAN MEMORY ALLOWANCE --- FOR TO USE SPARSE DICT
            elif self.data_mem_size_as_array > self.memory_cutoff and self.data_mem_size_as_sd <= self.memory_cutoff:
                print(f'\nDATA AS ARRAY ({self.data_size:,.0f} elements, est {self.data_mem_size_as_array:,.1f} MB) REQUIRES MORE MEMORY '
                      f'THAN SPARSE DICT ({self.number_of_data_elements:,.0f} elements, est {self.data_mem_size_as_sd:,.1f} MB). '
                      f'RUNNING AS SPARSE DICT.')
                self.run_data_as_sparse_dict = True
            # BOTH ARRAY & SD BELOW MEMORY ALLOWANCE --- ALLOW CHOICE FOR NOW, PROBABLY SOMEDAY WILL FORCE BASED ON SPARSITY
            elif self.data_mem_size_as_array <= self.memory_cutoff and self.data_mem_size_as_sd <= self.memory_cutoff:
                print(f'\nDATA AS ARRAY - '.ljust(19) + f'{self.data_size:,.0f} elements, {self.start_sparsity}% sparsity, est {self.data_mem_size_as_array:,.1f} MB')
                print(f'DATA AS SPARSE DICT - '.ljust(19) + f'{self.number_of_data_elements:,.0f} elements, est {self.data_mem_size_as_sd:,.1f} MB')
                self.run_data_as_sparse_dict = vui.validate_user_str(f'Run as ARRAY(a) or SPARSE DICT(s) > ', 'AS') == 'S'

                # if self.start_sparsity >= self.start_sparsity_cutoff:
                #     if self.data_mem_size_as_sd < self.data_mem_size_as_array:
                #         self.run_data_as_sparse_dict = True
                #     elif self.data_mem_size_as_sd >= self.data_mem_size_as_array:
                #         self.run_data_as_sparse_dict = False
                # elif self.start_sparsity < self.start_sparsity_cutoff:
                #     self.run_data_as_sparse_dict = False

        else:  # elif self.prompt_bypass:
            if self.run_data_as is None:
                if self.data_mem_size_as_array <= self.data_mem_size_as_sd:
                    print(f'\nMEMORY USAGE IS LOWER AS ARRAY, RUNNING AS ARRAY.')
                    self.run_data_as_sparse_dict = False
                else:
                    print(f'\nMEMORY USAGE IS LOWER AS SPARSE DICT, RUNNING AS SPARSE DICT.')
                    self.run_data_as_sparse_dict = True
            else:
                if self.run_data_as == 'ARRAY':
                    print(f'RUNNING DATA AS USER-SELECTED ARRAY.')
                    self.run_data_as_sparse_dict = False
                elif self.run_data_as == 'SPARSE_DICT':
                    print(f'RUNNING DATA AS USER-SELECTED SPARSE DICT.')
                    self.run_data_as_sparse_dict = True

        self.run_data_as = "SPARSE_DICT" if self.run_data_as_sparse_dict else "ARRAY"


    def build(self):

        ## 9/21/22 CREATE "LOOSE" RULE THAT AAT SPARSITY CANNOT BE GREATER THAN SPARSITY OF LEAST SPARSE FEATURE IN DATA
        print(f'\nPREDICTING STATISTICS OF KERNEL...')
        self.K_size = self._outer_len * self._outer_len
        # 9/27/22 ... SEPARATELY DEVELOPED A HEURSTIC FOR PREDICTING sparsity(K) FROM ATTRIBUTES OF DATA.  THE EQN IS:
        '''
        0.136145082	    FLOAT_COLUMNS
        0.841548882	    MIN_FLOAT_SP
        -0.138953628	MAX_FLOAT_SP
        0.09993363	    START_SPARSITY
        0.000183567	    AVG_CAT_SP_x_AVG_CAT_SP_x_MIN_FLOAT_SP
        -0.000479135	CAT_COLUMNS_x_MAX_CAT_SP_x_MIN_FLOAT_SP
        -0.000148671	MAX_CAT_SP_x_MIN_FLOAT_SP_x_MAX_CAT_SP
        2.37957E-05	    MIN_CAT_SP_x_START_SPARSITY_x_AVG_FLOAT_SP
        -3.04407E-05	MIN_FLOAT_SP_x_AVG_FLOAT_SP_x_MIN_FLOAT_SP
        -1.18105E-05	AVG_CAT_SP_x_FLOAT_COLUMNS_x_AVG_CAT_SP
        '''
        # K_pred_sparsity = 0.136145082 * self.data_ct_float_columns
        #                 + 0.841548882 * self.data_min_float_sparsity
        #                 - 0.138953628 * self.data_max_float_sparsity
        #                 + 0.09993363 * self.start_sparsity
        #                 + 0.000183567 * self.data_avg_cat_sparsity * self.data_avg_cat_sparsity * self.data_min_float_sparsity
        #                 - 0.000479135 * self.data_ct_cat_columns * self.data_max_cat_sparsity * self.data_min_float_sparsity
        #                 - 0.000148671 * self.data_max_cat_sparsity * self.data_min_float_sparsity * self.data_max_cat_sparsity
        #                 + 2.37957E-05 * self.data_min_cat_sparsity * self.start_sparsity * self.data_avg_float_sparsity
        #                 - 3.04407E-05 * self.data_min_float_sparsity * self.data_avg_float_sparsity * self.data_min_float_sparsity
        #                 - 1.18105E-05 * self.data_avg_cat_sparsity * self.data_ct_float_columns * self.data_avg_cat_sparsity

        self.K_pred_sparsity = round(min(100, max(0, 0.136145082 * self.data_ct_float_columns
                + 0.841548882 * self.data_min_float_sparsity
                - 0.138953628 * self.data_max_float_sparsity
                + 0.09993363 * self.start_sparsity
                + 0.000183567 * self.data_avg_cat_sparsity * self.data_avg_cat_sparsity * self.data_min_float_sparsity
                - 0.000479135 * self.data_ct_cat_columns * self.data_max_cat_sparsity * self.data_min_float_sparsity
                - 0.000148671 * self.data_max_cat_sparsity * self.data_min_float_sparsity * self.data_max_cat_sparsity
                + 2.37957E-05 * self.data_min_cat_sparsity * self.start_sparsity * self.data_avg_float_sparsity
                - 3.04407E-05 * self.data_min_float_sparsity * self.data_avg_float_sparsity * self.data_min_float_sparsity
                - 1.18105E-05 * self.data_avg_cat_sparsity * self.data_ct_float_columns * self.data_avg_cat_sparsity)), 2)

        self.K_pred_elements = round(self.K_size * (100 - self.K_pred_sparsity) / 100, 0)
        self.K_pred_mem_size_as_array = round(self.K_size * ms.MemSizes('np_float').mb(), 0)
        self.K_pred_mem_size_as_sd = round(self.K_pred_elements * ms.MemSizes('sd_float').mb(), 0)
        self.full_K_pred_mem_size_as_sd = round(self.K_size * ms.MemSizes('sd_float').mb(), 0)


        if self.prompt_bypass is False:
            print(f'\nKERNEL IS {self._outer_len} x {self._outer_len} WITH ESTIMATED SPARSITY OF {self.K_pred_sparsity}% '
                  f'BEFORE APPLYING KERNEL FUNCTION.')
            print(f'ESTIMATED KERNEL SIZE AS ARRAY WITH {self.K_size:,.0f} ELEMENTS IS {self.K_pred_mem_size_as_array:,.1f} MB')
            print(f'ESTIMATED KERNEL SIZE AS SPARSE DICT WITH EST {self.K_pred_elements:,.0f} ELEMENTS IS EST {self.K_pred_mem_size_as_sd:,.1f} MB')
            print(f'ESTIMATED KERNEL SIZE AS DENSE SPARSE DICT WITH {self.K_size:,.0f} ELEMENTS IS {self.full_K_pred_mem_size_as_sd:,.1f} MB')

            self.return_as = {'A':'ARRAY', 'S':'SPARSE_DICT'}[vui.validate_user_str(f'\nPRODUCE KERNEL AS ARRAY(a) OR SPARSE DICT(s) > ', 'AS')]

        else:  # elif self.prompt_bypass:
            if self.return_kernel_as is None:
                # ASSUME FULLY DENSE ARRAY IS ALWAYS LOWER MEMORY THAN FULLY DENSE SPARSE DICT
                if self.kernel_fxn != 'LINEAR':
                    print(f'\nKERNEL FUNCTION "{self.kernel_fxn}" IS POTENTIALLY 100% DENSE. RETURNING KERNEL AS ARRAY.')
                    self.return_as = 'ARRAY'
                else:
                    if self.K_pred_mem_size_as_array <= self.K_pred_mem_size_as_sd:
                        print(f'\nKERNEL ESTIMATES INDICATE LOWER MEMORY USAGE AS ARRAY. RETURNING KERNEL AS ARRAY.')
                        self.return_as = 'ARRAY'
                    else:
                        print(f'\nKERNEL ESTIMATES INDICATE LOWER MEMORY USAGE AS SPARSE DICT. RETURNING AS SPARSE DICT.')
                        self.return_as = 'SPARSE_DICT'
            else:
                print(f'\nRETURNING KERNEL AS USER-SELECTED {["ARRAY" if self.return_kernel_as == "ARRAY" else "SPARSE DICT"][0]}.')
                self.return_as = self.return_kernel_as

        self.return_kernel_as = self.return_as

        ##########################################################################################

        print(f'\nPREPARING DATA...')
        if self.is_list and self.run_data_as_sparse_dict:
            self.DATA = sd.zip_list(self.DATA)
            self.is_list, self.is_dict = False, True
        if self.is_dict and not self.run_data_as_sparse_dict:
            self.DATA = sd.unzip_to_ndarray(self.DATA)[0]
            self.is_list, self.is_dict = True, False
        if self.is_dict and self.run_data_as_sparse_dict: pass
        if self.is_list and not self.run_data_as_sparse_dict: pass
        print(f'Done.')

        # GET BASELINE MEM TO GET psutil ACTUAL MEM OF K
        baseline_mem = int(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)

        # BUILD K #####################################################################################################
        if self.is_list:
            if self.return_as == 'ARRAY':
                print(f'\nBUILDING KERNEL AS ARRAY FROM DATA AS ARRAY...')
                if self.kernel_fxn in ['LINEAR', 'POLYNOMIAL']:
                    print(f'\nRunning matmul...')
                    DOT_ARRAY = n.matmul(self.DATA.astype(float), self.DATA.transpose().astype(float), dtype=n.float64)
                    print(f'Done.')
                    if self.kernel_fxn == 'LINEAR':
                        self.K = sk.linear(DOT_ARRAY, return_as='ARRAY')
                        old_K_pred_mem_size_as_sd = self.K_pred_mem_size_as_sd
                    elif self.kernel_fxn == 'POLYNOMIAL':
                        self.K = sk.polynomial(DOT_ARRAY, self.constant, self.exponent, return_as='ARRAY')
                        if self.constant == 0: old_K_pred_mem_size_as_sd = self.K_pred_mem_size_as_sd
                        elif self.constant != 0: old_K_pred_mem_size_as_sd = self.full_K_pred_mem_size_as_sd
                elif self.kernel_fxn == 'GAUSSIAN':
                    self.K = sk.gaussian(self.DATA, self.sigma, return_as='ARRAY')
                    old_K_pred_mem_size_as_sd = self.full_K_pred_mem_size_as_sd
            elif self.return_as == 'SPARSE_DICT':
                print(f'\nBUILDING KERNEL AS SPARSE DICT FROM DATA AS ARRAY...')
                if self.kernel_fxn in ['LINEAR', 'POLYNOMIAL']:
                    SPARSE_DOT_DICT = sd.sparse_matmul_from_lists(self.DATA, self.DATA.transpose(), LIST2_TRANSPOSE=self.DATA, is_symmetric=True)
                    if self.kernel_fxn == 'LINEAR':
                        self.K = sk.sparse_linear(SPARSE_DOT_DICT, return_as='SPARSE_DICT')
                    elif self.kernel_fxn == 'POLYNOMIAL':
                        self.K = sk.sparse_polynomial(SPARSE_DOT_DICT, self.constant, self.exponent, return_as='SPARSE_DICT')
                        # K IS ALWAYS "FULLY SIZED" WHEN ARRAY, SO SIZE AS IS AND SIZE AS "FULL" ARE EQUAL
                elif self.kernel_fxn == 'GAUSSIAN':
                    self.K = sk.sparse_gaussian(self.DATA, self.DATA, self.sigma, return_as='SPARSE_DICT')

                old_K_pred_mem_size_as_array = self.K_pred_mem_size_as_array

        elif self.is_dict:
            if self.return_as == 'ARRAY':
                print(f'\nBUILDING KERNEL AS ARRAY FROM DATA AS SPARSE DICT...')
                if self.kernel_fxn in ['LINEAR', 'POLYNOMIAL']:
                    DOT_ARRAY = sd.core_symmetric_matmul(self.DATA, sd.sparse_transpose(self.DATA),
                                                               DICT2_TRANSPOSE=self.DATA, return_as='ARRAY')
                    if self.kernel_fxn == 'LINEAR':
                        self.K = sk.linear(DOT_ARRAY, return_as='ARRAY')
                        old_K_pred_mem_size_as_sd = self.K_pred_mem_size_as_sd
                    elif self.kernel_fxn == 'POLYNOMIAL':
                        self.K = sk.polynomial(DOT_ARRAY, self.constant, self.exponent, return_as='ARRAY')
                        if self.constant == 0: old_K_pred_mem_size_as_sd = self.K_pred_mem_size_as_sd
                        elif self.constant != 0: old_K_pred_mem_size_as_sd = self.full_K_pred_mem_size_as_sd
                elif self.kernel_fxn == 'GAUSSIAN':
                    self.K = sk.sparse_gaussian(self.DATA, self.DATA, self.sigma, return_as='ARRAY')
                    old_K_pred_mem_size_as_sd = self.K_pred_mem_size_as_sd
            elif self.return_as == 'SPARSE_DICT':
                print(f'\nBUILDING KERNEL AS SPARSE DICT FROM DATA AS SPARSE DICT...')
                if self.kernel_fxn in ['LINEAR', 'POLYNOMIAL']:
                    SPARSE_DOT_DICT = sd.core_symmetric_matmul(self.DATA, sd.sparse_transpose(self.DATA),
                                                               DICT2_TRANSPOSE=self.DATA, return_as='SPARSE_DICT')
                    if self.kernel_fxn == 'LINEAR':
                        self.K = sk.sparse_linear(SPARSE_DOT_DICT, return_as='SPARSE_DICT')
                    elif self.kernel_fxn == 'POLYNOMIAL':
                        self.K = sk.sparse_polynomial(SPARSE_DOT_DICT, self.constant, self.exponent, return_as='SPARSE_DICT')
                        # K IS ALWAYS "FULLY SIZED" WHEN ARRAY, SO SIZE AS IS AND SIZE AS "FULL" ARE EQUAL
                elif self.kernel_fxn == 'GAUSSIAN':
                    self.K = sk.sparse_gaussian(self.DATA, self.DATA, self.sigma, return_as='SPARSE_DICT')

                old_K_pred_mem_size_as_array = self.K_pred_mem_size_as_array

        print(f'Done.')

        mem = int(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2) - baseline_mem
        print(f'\n***** psutil Δ MEM FOR K: {mem} MB *****\n')

        # END BUILD K #####################################################################################################

        if self.prompt_bypass is False:
            # INFO GATHERING ######################################################################################################
            print(f'\nDisplaying info gathered about memory sizes of DATA and K as arrays and dicts...')
            if self.return_as == 'ARRAY':
                self.K_act_sparsity = round(sd.list_sparsity(self.K), 2)
                self.K_act_elements = round(len(self.DATA)**2 * (1 - self.K_act_sparsity / 100), 0)
                self.K_act_mem_size_as_array = round(len(self.DATA)**2 * ms.MemSizes('np_float').mb(), 1)
                self.K_pred_mem_size_as_sd = round(self.K_act_elements * ms.MemSizes('sd_float').mb(), 1)
            elif self.return_as == 'SPARSE_DICT':
                self.K_act_sparsity = round(sd.sparsity(self.K), 2)
                self.K_act_elements = round(len(self.DATA)**2 * (1 - self.K_act_sparsity / 100), 0)
                self.K_act_mem_size_as_sd = round(self.K_act_elements * ms.MemSizes('sd_float').mb(), 1)
                self.K_pred_mem_size_as_array = round(len(self.DATA)**2 * ms.MemSizes('np_float').mb(), 1)

            del self.DATA

            print(f'\nCOMPARE PREDICTED TO ACTUAL FOR K AS {self.return_as} (THE CURRENTLY RETURNED STATE):')
            if self.return_as == 'ARRAY':
                TABLE = [
                           [f'{self.K_pred_mem_size_as_array:,.1f}', f'{self.K_act_mem_size_as_array:,.1f}'],
                           [f'{self.K_pred_sparsity:,.2f}', f'{self.K_act_sparsity:,.2f}'],
                           [f'{self.K_pred_elements:,.0f}', f'{self.K_act_elements:,.0f}']
                        ]
            elif self.return_as == 'SPARSE_DICT':
                TABLE = [
                           [f'{self.K_pred_mem_size_as_sd:,.1f}', f'{self.K_act_mem_size_as_sd:,.1f}'],
                           [f'{self.K_pred_sparsity:,.2f}', f'{self.K_act_sparsity:,.2f}'],
                           [f'{self.K_pred_elements:,.0f}', f'{self.K_act_elements:,.0f}']
                        ]
            HEADER = ['PREDICTED', 'ACTUAL']

            print(
                p.DataFrame(
                    data=TABLE, columns=HEADER, index=['MEM SIZE(MB)', 'SPARSITY', 'EST ELEMENTS'], dtype=str
                )
            )

            print(f'\nOLD PREDICTED SIZE FOR K AS '
                  f'{["SPARSE DICT" if self.return_as=="ARRAY" else "ARRAY"][0]} (THE NOT-RETURNED STATE) WAS '
                  f'{[f"{old_K_pred_mem_size_as_sd:,.1f}" if self.return_as=="ARRAY" else f"{old_K_pred_mem_size_as_array:,.1f}"][0]}, NOW IS '
                  f'{[f"{self.K_pred_mem_size_as_sd:,.1f}" if self.return_as=="ARRAY" else f"{self.K_pred_mem_size_as_array:,.1f}"][0]}.')

            _ = input(f'\nHit ENTER to continue > ')
            # END INFO GATHERING ######################################################################################################

            print(f'\nCURRENTLY SET TO RETURN AS {self.return_as}. ')
            print(f'CURRENT SIZE AS {self.return_as} IS {str([self.K_act_mem_size_as_array if self.return_as=="ARRAY" else self.K_act_mem_size_as_sd][0])} MB. '
                  f'PROJECTED SIZE AS {["SPARSE DICT" if self.return_as=="ARRAY" else "ARRAY"][0]} IS '
                  f'{str([self.K_pred_mem_size_as_sd if self.return_as=="ARRAY" else self.K_pred_mem_size_as_array][0])} MB.')
            if vui.validate_user_str(f'RETURN AS ARRAY(a) OR SPARSE DICT(s)? > ', 'AS') == 'A': self.return_as = 'ARRAY'
            else: self.return_as = 'SPARSE_DICT'

        else:  # elif self.prompt_bypass:
            # REGARDLESS OF HOW self.return_kernel_as IS SET... THAT APPLIES WHEN BUILDING, THEN MEMORY DICTATES ULTIMATE RETURN
            if self.K_act_mem_size_as_array <= self.K_act_mem_size_as_sd:
                print(f'\nACTUAL KERNEL MEMORY USAGE IS LOWER AS ARRAY. RETURNING AS ARRAY.\n')
                self.return_as = 'ARRAY'
            else:
                print(f'\nACTUAL KERNEL MEMORY USAGE IS LOWER AS SPARSE DICT. RETURNING AS SPARSE DICT.\n')
                self.return_as = 'SPARSE_DICT'


        if self.return_as == 'ARRAY' and isinstance(self.K, n.ndarray):
            return self.return_fxn()
        elif self.return_as == 'SPARSE_DICT' and isinstance(self.K, n.ndarray):
            print(f'Zipping K TO SPARSE DICT...')
            self.K = sd.zip_list(self.K)
            print(f'Done.')
            return self.return_fxn()
        elif self.return_as == 'ARRAY' and isinstance(self.K, dict):
            self.K = sd.unzip_to_ndarray(self.K)[0]
            return self.return_fxn()
        elif self.return_as == 'SPARSE_DICT' and isinstance(self.K, dict):
            return self.return_fxn()


        print(f'\n...Kernel matrix complete.')



    def return_fxn(self):
        if self.prompt_bypass is True:
            return self.K
        else:
            return self.K, self.run_data_as, self.return_kernel_as









if __name__ == '__main__':
    import time, psutil, os

    baseline_mem = int(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)

    # DATA MUST GO IN AS [[] = ROWS]!
    rows = 10000
    cols = 100
    sparsity = 90

    # AS ARRAY #####################################################
    # DATA = n._random_.randint(0,10000,(rows,cols))
    #
    # cutoff = (1 - sparsity / 100) * 10000
    #
    # DATA = DATA * (DATA <= cutoff)
    #
    # # RESCALE BACK TO 0 -> 9
    # DATA = n.ceil(DATA / cutoff * 9)
    #
    # SPARSE_DICT = zip_list(DATA.astype(int))

    # AS DICT #####################################################
    DATA = sd.create_random(rows,cols,sparsity)

    time.sleep(5)
    mem = int(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2) - baseline_mem
    print(f'DATA OBJECT IS {mem} MB')

    K = KernelBuilder(DATA, kernel_fxn='LINEAR', constant=0, exponent=1, sigma=1, DATA_HEADER=None, prompt_bypass=True).build()








