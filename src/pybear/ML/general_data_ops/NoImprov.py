import numpy as n
from general_sound import winlinsound as wls
from data_validation import validate_user_input as vui, arg_kwarg_validater as akv


'''
    INSTRUCTIONS -- SET UP LOOP FOR IMPROVEMENT TRACKING LIKE THIS --- EXAMPLE FOR max() 
    
    external_ctr = 0 
    no_improv_ctr = float('-inf') 
    best_value = 10
    no_improv_limit = 100
    pct_change = 0.1
    
    for _ in range(1, 10000):

        **** PUT USER FUNCTION TO MONITOR HERE ****

        external_ctr += 1
        no_improv_ctr, no_improv_limit, best_value, abort = \
            NoImprov(value, external_ctr, 1, no_improv_ctr, no_improv_limit, best_value, 
                    pct_change, 'function_name', conv_end_method).max()
        if abort:
            break
'''


class NoImprov:

    def __init__(self, value, external_ctr, external_ctr_init, no_improv_ctr, no_improv_limit, best_value, pct_change,
                 function_name, conv_end_method='PROMPT'):

        self.value = float(value)
        self.external_ctr = int(external_ctr)
        self.external_ctr_init = int(external_ctr_init)

        # CAN BE ONLY BE "PROMPT" OR "KILL"
        self.conv_end_method = akv.arg_kwarg_validater(conv_end_method, 'conv_end_method', ['KILL', 'PROMPT', None],
                                                       'NoImprov', '__init__')

        if self.external_ctr == self.external_ctr_init:
            self.no_improv_ctr = -1  # +=1 BELOW, SO THAT AFTER FIRST PASS RETURNS 0, ALSO ALLOWS no_improv_ctr TO BE INITIALIZED IN
                                     # ANY WAY (as None, False, etc.) FROM CALLING MODULE AND WILL ALWAYS RETURN 0 FOR FIRST PASS
            self.best_value = value  # ALLOWS best_value TO BE INITIALIZED IN ANY WAY (as None, False, etc.) FROM CALLING MODULE
        else:
            self.no_improv_ctr = int(no_improv_ctr)
            self.best_value = best_value

        self.no_improv_limit = float(no_improv_limit)   # MUST ACCOMMODATE inf

        if self.no_improv_limit == 0:
            raise ValueError(f'\n*** INVALID no_improv_limit "{self.no_improv_limit}" IN NoImprov. CANNOT BE 0. ***\n')

        self.pct_change = float(pct_change)
        if self.pct_change < 0:
            raise ValueError(f'\n*** INVALID pct_change "{self.pct_change}" IN NoImprov. MUST BE >= 0. ***\n')

        self.function_name = function_name

        # HOLDERS
        self.formula = None
        self.abort = False



    def return_fxn(self):
        return self.no_improv_ctr, self.no_improv_limit, self.best_value, self.abort
        # RETURN no_improv_limit BECAUSE OF OPTION TO CHANGE ON THE FLY


    def core_engine(self):

        if self.conv_end_method is None:
            self.no_improv_ctr=0
            return self.return_fxn()

        # CORE IMPROVEMENT DECIDER ##################################################################################################

        if self.improvement_indicator > 0:

            with n.errstate(all='ignore'):

                # SHOULDNT HAVE TO WORRY ABT best_value infs HERE, SHOULD BE HANDLED BY best_value __init__

                if self.improvement_indicator / abs(self.best_value) * 100 > self.pct_change:
                    self.no_improv_ctr = 0
                    self.best_value = self.value
                else:
                    self.no_improv_ctr += 1

        elif self.improvement_indicator <= 0:
            self.no_improv_ctr += 1

        # END CORE IMPROVEMENT DECIDER ##############################################################################################

        if self.no_improv_ctr >= self.no_improv_limit:
            print(f'\n**** {self.function_name} TERMINATED FOR NO IMPROVEMENT. ****\n')

            if self.conv_end_method == 'PROMPT':
                wls.winlinsound(888, 500)
                user_select = vui.validate_user_str(f'end / accept result(e) ignore once(i) ignore always(a) change no improvement count limit(c) > ', 'EIAC')
            elif self.conv_end_method == 'KILL':
                user_select = 'E'

            if user_select == 'E':
                self.abort = True
            elif user_select in ['I', 'A', 'C']:
                self.no_improv_ctr = 0
                # if user_select == 'I': pass
                if user_select == 'A': self.no_improv_limit = float('inf')
                elif user_select == 'C': self.no_improv_limit = vui.validate_user_int(f'Enter new no improvement count limit > ', min=1)


    def max(self):
        # WANT IMPROVEMENT TO BE A POSITIVE NUMBER
        self.improvement_indicator = self.value - self.best_value
        self.core_engine()

        return self.return_fxn()


    def min(self):
        # WANT IMPROVEMENT TO BE A POSITIVE NUMBER
        self.improvement_indicator = self.best_value - self.value
        self.core_engine()

        return self.return_fxn()


    def two_sided_max(self):
        # WANT IMPROVEMENT TO BE A POSITIVE NUMBER
        self.improvement_indicator = abs(self.value) - abs(self.best_value)
        self.core_engine()

        return self.return_fxn()


    def two_sided_min(self):
        # WANT IMPROVEMENT TO BE A POSITIVE NUMBER
        self.improvement_indicator = abs(self.best_value) - abs(self.value)
        self.core_engine()

        return self.return_fxn()














if __name__ == '__main__':

    external_ctr_init, no_improv_ctr, best_value, no_improv_limit = 1, 0, float('-inf'), 1
    VALUE = []
    for _ in range(external_ctr_init,10000):
        value = n.log(_) + 125
        print(f'Trial {_} value = {value}')
        VALUE.append(value)

        no_improv_ctr, no_improv_limit, best_value, abort = \
            NoImprov(value, _, external_ctr_init, no_improv_ctr, no_improv_limit, best_value, .1, 'log',
                        conv_end_method = 'KILL').max()

        if abort: print(f'BREAK FOR NO IMPROVE ON ITR {_}'); break




















