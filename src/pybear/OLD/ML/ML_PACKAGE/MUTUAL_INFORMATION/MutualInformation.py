import sys
import numpy as np
from debug import get_module_name as gmn
from data_validation import arg_kwarg_validater as akv
from ML_PACKAGE._data_validation import list_dict_validater as ldv
from general_data_ops import get_shape as gs
from scipy.special import logsumexp
import sparse_dict as sd
from MLObjects import MLObject as mlo
from MLObjects.ObjectOrienter import MLObjectOrienter as mloo
from ML_PACKAGE.MUTUAL_INFORMATION import MICrossEntropyObjects as miceo



# DATA & TARGET ARE PROCESSED AS [] = COLUMN
class MutualInformation:
    '''DATA and TARGET both must be single columns.'''
    def __init__(self,
                 DATA,
                 data_given_orientation,
                 TARGET,
                 target_given_orientation,
                 data_run_format='ARRAY',
                 data_run_orientation='COLUMN',
                 target_run_format='ARRAY',
                 target_run_orientation='COLUMN',
                 DATA_UNIQUES=None,
                 TARGET_UNIQUES=None,
                 Y_OCCUR_HOLDER=None,
                 Y_SUM_HOLDER=None,
                 Y_FREQ_HOLDER=None,
                 bypass_validation=None,
         ):





        # DATA IS PREFERRED AS ARRAY AND [[]=COLUMN]
        # TARGET IS PREFERRED ARRAY AND [[]=COLUMN]
        # 9-14-22 CHANGED TO NDARRAYS BASED ON SPEED TESTS OF np.MATMUL VS sd.MATMUL
        # 11-7-22 HAD ORIGINALLY IMPLEMENTED THIS BY CHANGING dict HANDLING FOR Y_HOLDERS, ETC, BUT
        #                   REVERTING BACK TO OLD dict CODE & MANAGING CHANGE TO NDARRAYS VIA KWARGS & ObjectOrienter
        # 4-17-23 REVISITED AND KEEPING AS FORCE TARGET AND DATA TO ARRAY & COLUMN

        # SPACES FOR CONGRUENCE W MLRegression




        this_module = gmn.get_module_name(str(sys.modules[__name__]))
        fxn = '__init__'

        bypass_validation = akv.arg_kwarg_validater(bypass_validation, 'bypass_validation', [True, False, None],
                                                         this_module, fxn, return_if_none=False)

        if not bypass_validation:
            data_given_orientation = akv.arg_kwarg_validater(data_given_orientation, 'data_given_orientation', ['ROW', 'COLUMN'],
                                                             this_module, fxn)
            target_given_orientation = akv.arg_kwarg_validater(target_given_orientation, 'target_given_orientation', ['ROW', 'COLUMN'],
                                                             this_module, fxn)

            data_format, DATA = ldv.list_dict_validater(DATA, 'DATA')

            del data_format

            TARGET = ldv.list_dict_validater(TARGET, 'TARGET')[1]

            # VALIDATION OF NON-MULTICLASS TARGET IS HANDLED BY ObjectOrienter WHEN NOT BYPASS VALIDATION










        ########################################################################################################################
        # ORIENT DATA AND TARGET OBJECTS #######################################################################################





        # TARGET AND DATA ARE PROCESSED AS [[]=COLUMN] IN MutualInformation
        # 5/5/2023 --- FANCY if/else FOR TESTING MI SCORE ACCURACY UNDER DIFFERENT CONFIGS
        # BEAR
        # data_run_format='ARRAY' if data_run_format is None else data_run_format
        # data_run_orientation='COLUMN' if data_run_orientation is None else data_run_orientation



        # IF ALL THE Y_HOLDER OBJECTS ARE PASSED, DONT NEED TARGET SO DONT BOTHER TO CHANGE TARGET
        if True not in map(lambda x: x is None, (Y_OCCUR_HOLDER, Y_SUM_HOLDER, Y_FREQ_HOLDER)):
            target_run_format, target_run_orientation = 'AS_GIVEN', 'AS_GIVEN'
        # BEAR
        # else:
        #     target_run_format = 'ARRAY' if target_run_format is None else target_run_format
        #     target_run_orientation = 'COLUMN' if target_run_orientation is None else target_run_orientation

        OrienterClass = mloo.MLObjectOrienter(
                                                DATA=DATA,
                                                data_given_orientation=data_given_orientation,
                                                data_return_orientation=data_run_orientation,
                                                data_return_format=data_run_format,

                                                target_is_multiclass=False,
                                                TARGET=TARGET,
                                                target_given_orientation=target_given_orientation,
                                                target_return_orientation=target_run_orientation,
                                                target_return_format=target_run_format,

                                                RETURN_OBJECTS=['DATA', 'TARGET'],

                                                bypass_validation=bypass_validation,  # CATCHES MULTICLASS TARGET, CHECKS TRANSPOSES
                                                calling_module=this_module,
                                                calling_fxn=fxn
        )

        data_run_format = OrienterClass.data_return_format
        data_run_orientation = OrienterClass.data_return_orientation
        target_run_format = OrienterClass.target_return_format
        target_run_orientation = OrienterClass.target_return_orientation
        DATA = OrienterClass.DATA   #.reshape((1,-1))[0] if data_run_format=='ARRAY' else OrienterClass.DATA
        TARGET = OrienterClass.TARGET   #.reshape((1, -1))[0] if target_run_format == 'ARRAY' else OrienterClass.TARGET

        del OrienterClass



















        # END ORIENT DATA AND TARGET OBJECTS ###################################################################################
        ########################################################################################################################












        #########################################################################################################################
        # BUILD Y_OCCUR, Y_SUM, Y_FREQ ##########################################################################################

        # THESE MAY HAVE BEEN GIVEN AS KWARGS
        # VALIDATE BECAUSE MAY NOT HAVE BEEN CREATED BY MICrossEntropyObjects

        if not True in map(lambda x: x is None, (Y_OCCUR_HOLDER, Y_SUM_HOLDER, Y_FREQ_HOLDER)):
            # IF ALL MI TARGET CROSS-ENTROPY OBJECTS WERE PASSED AS KWARGS, BYPASS BUILDS
            self.Y_OCCUR_HOLDER, self.Y_SUM_HOLDER, self.Y_FREQ_HOLDER = Y_OCCUR_HOLDER, Y_SUM_HOLDER, Y_FREQ_HOLDER
            y_formats = ldv.list_dict_validater(self.Y_OCCUR_HOLDER, 'Y_OCCUR_HOLDER')[0]

            if not bypass_validation:
                if False in map(lambda x: type(x)==type(self.Y_OCCUR_HOLDER), (self.Y_SUM_HOLDER, self.Y_FREQ_HOLDER)):
                    raise Exception(f'self.Y_OCCUR_HOLDER, self.Y_SUM_HOLDER, self.Y_FREQ_HOLDER MUST BE PASSED IN THE SAME FORMAT')

        else:  # if True in map(lambda x: x is None, (Y_OCCUR_HOLDER, Y_SUM_HOLDER, Y_FREQ_HOLDER)):
            # IF A MI TARGET CROSS-ENTROPY OBJECT(S) NEED TO BE BUILT

            if not bypass_validation:
                if gs.get_shape('TARGET', TARGET, target_run_orientation)[0] != \
                        gs.get_shape('DATA', DATA, data_run_orientation)[0]:
                    raise Exception(f'\n*** MutualInformation() TARGET len IS NOT EQUAL TO DATA len ***\n')

            if TARGET_UNIQUES is None:
                TargetUniquesClass = mlo.MLObject(TARGET,
                                                  target_run_orientation,
                                                  name='TARGET',
                                                  return_orientation='AS_GIVEN',
                                                  return_format='AS_GIVEN',
                                                  bypass_validation=bypass_validation,
                                                  calling_module=this_module, calling_fxn=fxn)

                TARGET_UNIQUES = TargetUniquesClass.unique(0).reshape((1,-1))
                del TargetUniquesClass


            # IF SIZE Y_OCCUR_HOLDER WOULD BE OVER 100,000,000 IF ARRAY, RETURN AS SPARSE_DICT
            y_formats = target_run_format if len(TARGET_UNIQUES[0]) * gs.get_shape('TARGET', TARGET, target_run_orientation)[0] < 1e8 else "SPARSE_DICT"

            if Y_OCCUR_HOLDER is None:
                self.Y_OCCUR_HOLDER = miceo.occurrence(TARGET, OBJECT_UNIQUES=TARGET_UNIQUES, return_as=y_formats,
                                bypass_validation=bypass_validation, calling_module=this_module, calling_fxn=fxn)
            else: self.Y_OCCUR_HOLDER = Y_OCCUR_HOLDER

            if Y_SUM_HOLDER is None: self.Y_SUM_HOLDER = miceo.sums(self.Y_OCCUR_HOLDER, return_as=y_formats,
                                                                     calling_module=this_module, calling_fxn=fxn)
            else: self.Y_SUM_HOLDER = Y_SUM_HOLDER

            if Y_FREQ_HOLDER is None: self.Y_FREQ_HOLDER = miceo.frequencies(self.Y_SUM_HOLDER,
                                    return_as=y_formats, calling_module=this_module, calling_fxn=fxn)
            else: self.Y_FREQ_HOLDER = Y_FREQ_HOLDER

        # DO THIS IN ALL CASES - IF SOME OR ALL OBJECTS WERE PASSED, VERIFY AGAINST ANYTHING NEWLY BUILD.  IF ALL ARE BUILT
        # ABOVE BY MICEO THEN THIS SHOULD BE A PROBLEM, MICEO IS TRUSTED
        if not bypass_validation:
            y_sum_cols = gs.get_shape('Y_SUM_HOLDER', self.Y_SUM_HOLDER, 'ROW')[1]
            y_freq_cols = gs.get_shape('Y_FREQ_HOLDER', self.Y_FREQ_HOLDER, 'ROW')[1]
            if y_sum_cols != y_freq_cols:
                raise Exception(f'*** Y_SUM_HOLDER ({y_sum_cols}) AND Y_FREQ_HOLDER ({y_freq_cols}) ARE NOT EQUAL SIZE ***')
            y_occ_cols = gs.get_shape('Y_OCCUR_HOLDER', self.Y_OCCUR_HOLDER, 'COLUMN')[1]
            if y_occ_cols != y_sum_cols:
                raise Exception(f'*** Y_OCCUR_HOLDER ({y_occ_cols}) IS NOT SAME SIZE AS Y_SUM AND Y_FREQ ({y_sum_cols}) ***')

            del y_sum_cols, y_freq_cols, y_occ_cols

        # END BUILD Y_OCCUR, Y_SUM, Y_FREQ ######################################################################################
        #########################################################################################################################

        #########################################################################################################################
        # BUILD X_OCCUR, X_SUM, X_FREQ ##########################################################################################


        if DATA_UNIQUES is None:
            DataUniquesClass = mlo.MLObject(DATA,
                                            data_run_orientation,
                                            name='DATA',
                                            return_orientation=data_run_orientation, # 'AS_GIVEN',
                                            return_format=data_run_format, # 'AS_GIVEN',
                                            bypass_validation=bypass_validation,
                                            calling_module=this_module, calling_fxn=fxn)

            DATA_UNIQUES = DataUniquesClass.unique(0).reshape((1, -1))
            del DataUniquesClass




        # IF SIZE X_OCCUR_HOLDER WOULD BE OVER 100,000,000 IF ARRAY, RETURN AS SPARSE_DICT
        x_formats = data_run_format if len(DATA_UNIQUES[0]) * gs.get_shape('DATA', DATA, data_run_orientation)[0] < 1e8 else "SPARSE_DICT"

        # THESE CANNOT BE PASSED AS KWARGS AND MUST BE BUILT EVERY INSTANTIATION
        # NO VALIDATION BECAUSE CREATED BY MICrossEntropyObjects

        XObjsClass = miceo.MICrossEntropyObjects(DATA, UNIQUES=DATA_UNIQUES, return_as=x_formats,
                                                 bypass_validation=bypass_validation
        )

        self.X_OCCUR_HOLDER = XObjsClass.OCCURRENCES
        self.X_SUM_HOLDER = XObjsClass.SUMS
        self.X_FREQ_HOLDER = XObjsClass.FREQ

        # END BUILD X_OCCUR, X_SUM, X_FREQ ######################################################################################
        #########################################################################################################################

        self.total_score = 0

        # (X OR Y)_OCCUR_HOLDER, (X OR Y)_SUM_HOLDER, (X OR Y)_FREQ_HOLDER COULD BE LIST OR DICT
        # Y MAYBE ARRAY OR DICT IF PASSED AS KWARG, BUT IS target_run_format IF NOT PASSED
        # X IS CREATED IN data_run_format

        for x_idx in range(len(self.X_OCCUR_HOLDER)):
            for y_idx in range(len(self.Y_OCCUR_HOLDER)):
                if x_formats=='ARRAY':
                    if y_formats=='ARRAY':
                        p_x_y = np.matmul(self.X_OCCUR_HOLDER[x_idx].astype(np.float64),
                                         self.Y_OCCUR_HOLDER[y_idx].astype(np.float64),
                                         dtype=np.float64
                        ) / self.Y_SUM_HOLDER[0][y_idx].astype(np.float64)
                    elif y_formats=='SPARSE_DICT':
                        p_x_y = np.sum(sd.core_hybrid_matmul(self.X_OCCUR_HOLDER[x_idx].astype(np.float64),
                                                            sd.core_sparse_transpose({0: self.Y_OCCUR_HOLDER[y_idx]}),
                                                            return_as='ARRAY',
                                                            return_orientation='COLUMN')
                        ) / self.Y_SUM_HOLDER[0][y_idx]

                elif x_formats=='SPARSE_DICT':
                    if y_formats=='ARRAY':
                        p_x_y = np.sum(sd.core_hybrid_matmul({0: self.X_OCCUR_HOLDER[x_idx]},
                                                            self.Y_OCCUR_HOLDER[y_idx].transpose(),
                                                            return_as='ARRAY',
                                                            return_orientation='COLUMN')
                        ) / self.Y_SUM_HOLDER[0][y_idx].astype(np.float64)
                    elif y_formats=='SPARSE_DICT':
                        p_x_y = sd.sum_(sd.core_matmul({0: self.X_OCCUR_HOLDER[x_idx]},
                                                  sd.core_sparse_transpose({0: self.Y_OCCUR_HOLDER[y_idx]}))
                        ) / self.Y_SUM_HOLDER[0][y_idx]

                try:
                    # 4-19-22 CONVERT TO logsumexp FOR DEALING WITH EXTREMELY BIG OR SMALL NUMBERS, GIVING RuntimeWarning AS A REGULAR FXN
                    # IF p_x_y GOES TO 0 (NUMERATOR IN LOG10) THEN BLOWUP, SO except AND pass IF p_x_y IS 0
                    with np.errstate(divide='raise'):
                        self.total_score += p_x_y * (np.log10(logsumexp(p_x_y)) - np.log10(logsumexp(self.X_FREQ_HOLDER[0][x_idx])) -
                                                     np.log10(logsumexp(self.Y_FREQ_HOLDER[0][y_idx])))
                except:
                    if RuntimeWarning or FloatingPointError: pass
                    
        del x_formats, y_formats

    # END init #############################################################################################################
    ########################################################################################################################
    ########################################################################################################################


    def run(self):
        return self.total_score







if __name__ == '__main__':
    import time
    from general_sound import winlinsound as wls

    # TEST FOR FUNCTIONALITY OF CODE AND ACCURACY OF MI SCORES

    # TEST CODE & MODULE VERIFIED GOOD 5/10/23

    def KILL(name, _format=None, ACT_OBJ=None, EXP_OBJ=None, act=None, exp=None):
        if not _format is None and not ACT_OBJ is None and not EXP_OBJ is None:
            if (_format=='ARRAY' and not np.array_equiv(ACT_OBJ, EXP_OBJ)) or (_format=='SPARSE_DICT' and not sd.core_sparse_equiv(ACT_OBJ, EXP_OBJ)):
                wls.winlinsound(444, 1000)
                print(f'\nACTUAL {name}:'); print(ACT_OBJ); print(f'\nEXPECTED {name}:'); print(EXP_OBJ)
                raise Exception(f'*** {name} ACTUAL AND EXPECTED NOT EQUAL ***')
        elif not act is None and not exp is None:
            wls.winlinsound(444, 1000)
            raise Exception(f'*** ACTUAL {name} ({act}) DOES NOT EQUAL EXPECTED ({exp}) ***')
        else:
            wls.winlinsound(444, 1000)
            raise Exception(f'*** YOU SCREWED UP THE EXCEPTION HANDLER!!!! ****')


    SKIPPED = [[], [], []]
    TABLE_OF_RESULTS = [[], [], []]

    number_of_data_categories = 4     # DATA UNIQUES ARE [0,1,2,3]
    number_of_target_categories = 2   # TARGET UNIQUES ARE [0,1]
    rows = 100

    # OBJECTS ARE BUILT AS [[]=COLUMN]
    TARGET_BASE_SPARSE_DICT = {0:{idx:[0,0,1][idx % 3] for idx in range(rows)}}
    TARGET_BASE_NUMPY = sd.unzip_to_ndarray_float64(TARGET_BASE_SPARSE_DICT)[0].reshape((1,-1))

    DATA_BASE_SPARSE_DICT = {0:{idx:idx % number_of_data_categories for idx in range(rows)}}
    DATA_BASE_NUMPY = sd.unzip_to_ndarray_float64(DATA_BASE_SPARSE_DICT)[0].reshape((1,-1))

    NP_EXP_X_OCCUR = np.vstack((DATA_BASE_NUMPY[0]==0, DATA_BASE_NUMPY[0]==1, DATA_BASE_NUMPY[0]==2, DATA_BASE_NUMPY[0]==3)).astype(np.int8)
    NP_EXP_X_SUMS = np.sum(NP_EXP_X_OCCUR, axis=1).reshape((1,-1))
    NP_EXP_X_FREQ = np.array(NP_EXP_X_SUMS/np.sum(NP_EXP_X_SUMS)).reshape((1,-1))
    SD_EXP_X_OCCUR = sd.zip_list_as_py_float(NP_EXP_X_OCCUR)
    SD_EXP_X_SUMS = sd.zip_list_as_py_float(NP_EXP_X_SUMS)
    SD_EXP_X_FREQ = sd.zip_list_as_py_float(NP_EXP_X_FREQ)

    NP_EXP_Y_OCCUR = np.vstack((TARGET_BASE_NUMPY[0]==0, TARGET_BASE_NUMPY[0]==1)).astype(np.int8)
    NP_EXP_Y_SUMS = np.sum(NP_EXP_Y_OCCUR, axis=1).reshape((1,-1))
    NP_EXP_Y_FREQ = np.array(NP_EXP_Y_SUMS/np.sum(NP_EXP_Y_SUMS)).reshape((1,-1))
    SD_EXP_Y_OCCUR = sd.zip_list_as_py_float(NP_EXP_Y_OCCUR)
    SD_EXP_Y_SUMS = sd.zip_list_as_py_float(NP_EXP_Y_SUMS)
    SD_EXP_Y_FREQ = sd.zip_list_as_py_float(NP_EXP_Y_FREQ)

    exp_score = 0
    for x_idx in range(len(NP_EXP_X_OCCUR)):
        for y_idx in range(len(NP_EXP_Y_OCCUR)):
            p_x_y = np.matmul(NP_EXP_X_OCCUR[x_idx].astype(np.float64), NP_EXP_Y_OCCUR[y_idx].astype(np.float64),
                             dtype=np.float64) / NP_EXP_Y_SUMS[0][y_idx].astype(np.float64)

            exp_score += p_x_y * (np.log10(logsumexp(p_x_y)) - np.log10(logsumexp(NP_EXP_X_FREQ[0][x_idx])) -
                                            np.log10(logsumexp(NP_EXP_Y_FREQ[0][y_idx])))






    ctr = 0

    for data_orientation in ['not transposed', 'transposed']:
        for data_stack_type in ['single', 'double']:
            for target_type in ['SPARSE_DICT', 'NUMPY']:
                for target_orientation in ['not transposed', 'transposed']:
                    for target_stack_type in ['single', 'double']:
                        for data_type in ['SPARSE_DICT', 'NUMPY']:
                            for data_run_format in ['ARRAY', 'SPARSE_DICT']:             # THESE WILL CONTROL THE FORMAT
                                for data_run_orientation in ['ROW', 'COLUMN']:           # OF (X OR Y)_OCCUR, (X OR Y)_SUMS,
                                    for target_run_format in ['ARRAY', 'SPARSE_DICT']:   # (X OR Y)_FREQ [EXCEPT FOR LARGE OCCUR ARRAY,
                                        for target_run_orientation in ['ROW', 'COLUMN']: # SEE MODULE FOR x_format, y_format RULES

                                            ctr += 1
                                            data_given_orientation = 'COLUMN'
                                            target_given_orientation = 'COLUMN'
                                            print(f'Running trial {ctr} of {2**10}')

                                            trial_name = f'DAT={data_stack_type} {"NP" if data_type=="NUMPY" else "SD"} as {"ROW" if data_orientation=="transposed" else "COL"}, ' \
                                                         f'TAR={target_stack_type} {"NP" if target_type=="NUMPY" else "SD"} as {"ROW" if target_orientation=="transposed" else "COL"}, ' \
                                                         f'data_run_format/orient={data_run_format}/{data_run_orientation}, target_run_format/orient={target_run_format}/{target_run_orientation}'
                                            print(f'{trial_name}')

                                            if data_orientation == 'transposed' and data_stack_type == 'single' or \
                                                target_orientation == 'transposed' and target_stack_type == 'single':   # TRANSPOSED CAN ONLY BE DOUBLE
                                                SKIPPED[0].append(trial_name)
                                                SKIPPED[1].append('skip')
                                                SKIPPED[2].append('skip')
                                                continue

                                            TABLE_OF_RESULTS[0].append(trial_name)

                                            if data_type == 'SPARSE_DICT': DATA = DATA_BASE_SPARSE_DICT
                                            elif data_type == 'NUMPY': DATA = DATA_BASE_NUMPY

                                            if data_orientation == 'transposed':
                                                if data_type == 'SPARSE_DICT': DATA = sd.sparse_transpose(DATA)
                                                elif data_type == 'NUMPY': DATA = DATA.transpose()
                                                data_given_orientation = 'ROW'
                                            # elif 'not transposed' stays the same

                                            if data_stack_type == 'single':
                                                if data_type == 'SPARSE_DICT': DATA = DATA[0]
                                                elif data_type == 'NUMPY': DATA = DATA[0]
                                            # elif 'double' stays the same



                                            if target_type == 'SPARSE_DICT': TARGET = TARGET_BASE_SPARSE_DICT
                                            elif target_type == 'NUMPY': TARGET = TARGET_BASE_NUMPY

                                            if target_orientation == 'transposed':
                                                if target_type == 'SPARSE_DICT': TARGET = sd.sparse_transpose(TARGET)
                                                elif target_type == 'NUMPY': TARGET = TARGET.transpose()
                                                target_given_orientation = 'ROW'
                                            # elif 'not transposed' stays the same

                                            if target_stack_type == 'single':
                                                if target_type == 'SPARSE_DICT': TARGET = TARGET[0]
                                                elif target_type == 'NUMPY': TARGET = TARGET[0]
                                            # elif 'double' stays the same
                                            t0 = time.time()
                                            TestClass = MutualInformation(DATA,
                                                                          data_given_orientation,
                                                                          TARGET,
                                                                          target_given_orientation,
                                                                          data_run_format=data_run_format,
                                                                          data_run_orientation=data_run_orientation,
                                                                          target_run_format=target_run_format,
                                                                          target_run_orientation=target_run_orientation,
                                                                          bypass_validation=False
                                                                          )

                                            act_score = TestClass.run()

                                            _time = time.time() - t0
                                            TABLE_OF_RESULTS[1].append(f'{act_score:,.3f}')
                                            TABLE_OF_RESULTS[2].append(f'{_time:,.2f}')

                                            ACT_X_OCCUR = TestClass.X_OCCUR_HOLDER
                                            ACT_X_SUMS = TestClass.X_SUM_HOLDER
                                            ACT_X_FREQ = TestClass.X_FREQ_HOLDER
                                            ACT_Y_OCCUR = TestClass.Y_OCCUR_HOLDER
                                            ACT_Y_SUMS = TestClass.Y_SUM_HOLDER
                                            ACT_Y_FREQ = TestClass.Y_FREQ_HOLDER


                                            print(f'\033[91m')
                                            if data_run_format=='ARRAY':
                                                KILL('X_OCCUR', _format='ARRAY', ACT_OBJ=ACT_X_OCCUR, EXP_OBJ=NP_EXP_X_OCCUR)
                                                KILL('X_SUMS', _format='ARRAY', ACT_OBJ=ACT_X_SUMS, EXP_OBJ=NP_EXP_X_SUMS)
                                                KILL('X_FREQ', _format='ARRAY', ACT_OBJ=ACT_X_FREQ, EXP_OBJ=NP_EXP_X_FREQ)
                                            elif data_run_format=='SPARSE_DICT':
                                                KILL('X_OCCUR', _format='SPARSE_DICT', ACT_OBJ=ACT_X_OCCUR, EXP_OBJ=SD_EXP_X_OCCUR)
                                                KILL('X_SUMS', _format='SPARSE_DICT', ACT_OBJ=ACT_X_SUMS, EXP_OBJ=SD_EXP_X_SUMS)
                                                KILL('X_FREQ', _format='SPARSE_DICT', ACT_OBJ=ACT_X_FREQ, EXP_OBJ=SD_EXP_X_FREQ)
                                            if target_run_format=='ARRAY':
                                                KILL('Y_OCCUR', _format='ARRAY', ACT_OBJ=ACT_Y_OCCUR, EXP_OBJ=NP_EXP_Y_OCCUR)
                                                KILL('Y_SUMS', _format='ARRAY', ACT_OBJ=ACT_Y_SUMS, EXP_OBJ=NP_EXP_Y_SUMS)
                                                KILL('Y_FREQ', _format='ARRAY', ACT_OBJ=ACT_Y_FREQ, EXP_OBJ=NP_EXP_Y_FREQ)
                                            elif target_run_format=='SPARSE_DICT':
                                                KILL('Y_OCCUR', _format='SPARSE_DICT', ACT_OBJ=ACT_Y_OCCUR, EXP_OBJ=SD_EXP_Y_OCCUR)
                                                KILL('Y_SUMS', _format='SPARSE_DICT', ACT_OBJ=ACT_Y_SUMS, EXP_OBJ=SD_EXP_Y_SUMS)
                                                KILL('Y_FREQ', _format='SPARSE_DICT', ACT_OBJ=ACT_Y_FREQ, EXP_OBJ=SD_EXP_Y_FREQ)

                                            if act_score != exp_score:
                                                KILL('score', act=act_score, exp=exp_score)

                                            print(f'\033[0m')

                                            # print(f'DATA = ')
                                            # print(DATA)
                                            # print(f'TARGET = ')
                                            # print(TARGET)
                                            # print(f'ACT_SCORE = ')
                                            # print(act_score)
                                            # print()
                                            # print()
                                            # time.sleep(3)

    import pandas as p
    p.set_option('display.max_columns', None, 'display.width', 140, 'display.max_rows', None)

    print(f'\nSKIPPED:')
    print(p.DataFrame(data=np.array(SKIPPED).transpose(), columns=['TEST', 'ACT_SCORE', 'TIME']))
    print()
    print(f'\nRESULTS:')
    print(p.DataFrame(data=np.array(TABLE_OF_RESULTS).transpose(), columns=['TEST', 'ACT_SCORE', 'TIME']))


    print(f'\033[92m\n*** ALL FUNCTIONALITY & ACCURACY TESTS PASSED ***\n\033[0m')
    for _ in range(3): wls.winlinsound(888, 500); time.sleep(1)














