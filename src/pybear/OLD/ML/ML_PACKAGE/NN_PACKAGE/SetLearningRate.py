import numpy as n
from data_validation import validate_user_input as vui
from general_list_ops import manual_num_seq_list_fill as mnslf, decay_fxn_seq_list_fill as dfslf

# return_fxn()
# learning_rate_calc()                  # not in use yet, for Cauchy, etc.
# measure()                             # checks state of LEARNING_RATE, if its the correct size
# print()                               # displays LEARNING_RATE(S)
# lr_user_entry()                       # shortcut used in several places for user manual entry of one learning rate
# set_learning_rate_all_same(self):     # set learning rate for all nodes simultaneously
# set_learning_rate_all_unique()        # set learning rate for all nodes separately
# set_learning_rate_one()               # set learning rate for one node
# set_learning_rate_calc()              # not in use yet
# learning_rate_run()                   # exe


class SetLearningRate:

    def __init__(self, LEARNING_RATE, ARRAY_OF_NODES, SELECT_LINK_FXN, lr_method, gd_iterations):
        self.LEARNING_RATE = LEARNING_RATE
        self.ARRAY_OF_NODES = ARRAY_OF_NODES
        self.SELECT_LINK_FXN = SELECT_LINK_FXN
        self.nodes = len(ARRAY_OF_NODES)
        self.lr_method = lr_method    # CAN BE 'C' OR 'S'   (CONSTANT OR CAUCHY)
        self.gd_iterations = gd_iterations
        self.node_idx = 0
        self.is_full, self.incorrect_full, self.empty = False, False, False


    def return_fxn(self):
        # BEAR 2-10-22 NOT IN USE, FOR SOME REASON IS RETURNING 'None' WHEN CALLED BELOW IN learning_rate_run
        # BUT DIRECT RETURN OF self.LEARNING_RATE WORKS (??)
        return self.LEARNING_RATE


    def learning_rate_calc(self):     #, BATCH_MATRIX, NEW_TARGET_VECTOR, ARRAY_OF_GRADIENTS, SELECT_LINK_FXN, iteration):
        # I THINK "iteration" IS IN THERE FOR POTENTIAL FUTURE CAPABILITY OF   learning_rate = f(iterations)
        link_fxn = self.SELECT_LINK_FXN[self.node_idx]

        if self.lr_method == 'C':
            pass

        elif self.lr_method == 'S' and link_fxn == 'None':
            print(f'CAUCHY METHOD FOR LEARNING RATE UNAVAILABLE.  MUST USE ANOTHER CONFIG.')
            # LEAST SQUARES PLACEHOLDER

            # RELICS OF LONG LOST ATTEMPTS TO CALCULATE LEARNING RATE FOR "LEAST SQUARES NO LINK" USING CAUCHY METHOD
            # learning_rate = n.divide(
            # n.sum(n.matmul(ARRAY_OF_NODES[0], X, dtype=float)) - n.sum(NEW_TARGET_VECTOR),
            # n.sum(n.matmul(UPDATE_VECTOR, X,  dtype=float))
            # )

            # LOG LIKELIHOOD PLACE HOLDERS

            # 9-6-2021 I'M THINKING THAT "S" IS GOING TO DEPEND ON LINK FXN, IN ADDITION TO COST FXN


    def measure(self):
        self.is_full = len(self.LEARNING_RATE)==len(self.ARRAY_OF_NODES) and \
                       False not in [self.gd_iterations==len(_) for _ in self.LEARNING_RATE]
        self.incorrect_full = len(self.LEARNING_RATE)>0 and (len(self.LEARNING_RATE)!=len(self.ARRAY_OF_NODES) or \
                         False in [len(_)==self.gd_iterations for _ in self.LEARNING_RATE])
        self.is_empty = len(self.LEARNING_RATE)==0 or (len(self.LEARNING_RATE)>0 and \
                        False not in [len(_)==0 for _ in self.LEARNING_RATE])
        if self.is_full:
            print(f'\nLEARNING_RATE is full and correctly sized with {len(self.LEARNING_RATE)} sequence(s) ' + \
                  f'of {len(self.LEARNING_RATE[0])} rates.')
        elif self.incorrect_full:
            print(f'\nLEARNING_RATE is incorrectly sized with {len(self.LEARNING_RATE)} rate sequence(s) (should be ' + \
                  f'{self.nodes} sequences of {self.gd_iterations} iterations).')
        elif self.is_empty:
            print(f'\nLEARNING_RATE is empty.')


    def print(self):
        LR_TEXT = [f':{len(LR_LIST)}' for LR_LIST in self.LEARNING_RATE]
        print(f'\nFIRST TEN LEARNING RATES (OF {", ".join(LR_TEXT)})')
        DUM_LR_VECTOR = [[f'{_:.5g}' for _ in __] for __ in self.LEARNING_RATE]
        [print(f'NODE {idx}) {DUM_LR_VECTOR[idx][:10]}') for idx in range(len(self.LEARNING_RATE))]
        print('')


    def lr_user_entry(self):
        fill_method =  vui.validate_user_str(f'Set all iterations the same(s) manual list fill(m) or decay function fill(d) > ', 'SMD')
        # SO ALL OBJECTS RETURNED IS [] OF LEN self.gd_iterations
        if fill_method == 'S':
            _ = vui.validate_user_float(f'Enter learning rate for all iterations > ', min=1e-20, max=1e3)
            return [_ for __ in range(self.gd_iterations)]
        elif fill_method == 'M':
            return mnslf.manual_num_seq_list_fill(f'number', [], self.gd_iterations, min=1e-30, max=1)
        elif fill_method == 'D':
            return dfslf.decay_fxn_seq_list_fill(self.gd_iterations, [])


    def set_learning_rate_all_same(self):
        while True:

            self.print()
            RATE_LIST = self.lr_user_entry()
            self.LEARNING_RATE = [RATE_LIST for _ in range(self.nodes)]
            if vui.validate_user_str(f'\nAccept learning rates? (y/n) > ', 'YN') == 'Y': break


    def set_learning_rate_all_unique(self):
        while True:

            self.print()
            self.LEARNING_RATE = [[] for _ in range(self.nodes)]

            for node_idx in range(self.nodes):
                print(f'\nNODE{node_idx}')
                self.LEARNING_RATE[node_idx] = self.lr_user_entry()

            if vui.validate_user_str(f'\nAccept learning rates? (y/n) > ', 'YN') == 'Y': break


    def set_learning_rate_one(self):
        while True:

            node_idx = vui.validate_user_int(f'Select node > ', min=0, max=self.nodes-1)
            self.LEARNING_RATE[node_idx] = self.lr_user_entry()

            self.print()

            if vui.validate_user_str(f'\nAccept learning rates? (y/n) > ', 'YN') == 'Y': break


    def set_learning_rate_calc(self):
        pass
        # 2-7-22 NOT SURE IF THIS IS NEEDED, THINKING CAUCHY WILL BE DONE INSIDE NNRun
        # self.LEARNING_RATE = lrc.learning_rate_calc(lr_method, LEARNING_RATE, node_idx, BATCH_MATRIX,
        #                       NEW_TARGET_VECTOR, ARRAY_OF_NODES, ARRAY_OF_GRADIENTS, SELECT_LINK_FXN, iteration)
        #                       PLACEHOLDER FOR CAUCHY CALCS, ETC FROM learning_rate_calc()


    def learning_rate_run(self):
        while True:

            self.measure()
            self.print()

            if self.lr_method == 'C':
                if self.is_empty or self.incorrect_full:   # IF EMPTY OR INCORRECT FULL, MUST SET ALL
                    _ = vui.validate_user_str(f'\nSet same LEARNING RATE for all nodes(s), enter individually(i), accept current(a) > ', 'SIA')
                elif self.is_full:  # IF FULL, CAN CHOOSE TO DO ALL OR ONLY ONE
                    _ = vui.validate_user_str(
                        f'\nSet LEARNING RATE for one node(o), set all nodes the same(s), set nodes individually(i), accept current(a) > ', 'OSIA')

                if _ == 'I': self.set_learning_rate_all_unique()
                elif _ == 'S': self.set_learning_rate_all_same()
                elif _ == 'O':
                    while True:
                        self.set_learning_rate_one()
                        if vui.validate_user_str(f'\nChange another? (y/n) > ', 'YN') == 'N':
                            break
                elif _ == 'A': pass

            elif self.lr_method == 'S':
                self.set_learning_rate_calc()
                # 2-7-22 NOT SURE IF THIS IS NEEDED, THINKING CAUCHY WILL BE DONE INSIDE NNRun

            self.measure()
            self.print()

            if vui.validate_user_str(f'\nAccept learning rate config? (y/n) > ', 'YN') == 'Y':
                break


        return self.LEARNING_RATE
        # self.return_fxn()  BEAR




if __name__ == '__main__':
    pass





































