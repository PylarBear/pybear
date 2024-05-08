import inspect
import os, sys, time
sys.tracebacklimit = None
import numpy as np
import sparse_dict as sd
from copy import deepcopy
from datetime import datetime
from debug import get_module_name as gmn
from general_sound import winlinsound as wls
from data_validation import validate_user_input as vui
from general_data_ops import NoImprov as ni, numpy_math as nm, get_shape as gs, new_np_random_choice as nnrc
from ML_PACKAGE.GENERIC_PRINT import show_time as st

from MLObjects import MLObject as mlo
from ML_PACKAGE.NN_PACKAGE import SetLearningRate as lrs
from ML_PACKAGE.NN_PACKAGE.link_functions import link_fxns as lf
from ML_PACKAGE.NN_PACKAGE.gd_run import generate_hessian as gh, cd_engine as ce, error_calc as ec, \
    output_vector_calc as ovc, gd_gd as gdgd, gd_rmsprop as gdr, gd_adam as gda
from ML_PACKAGE.NN_PACKAGE.print_results import NNSummaryStatistics as nnss



class NS(nm.NumpyMath):
    pass

class NNCoreRunCode:
    def __init__(self, DATA, TARGET_VECTOR, data_run_orientation, target_run_orientation, ARRAY_OF_NODES, OUTPUT_VECTOR,
                 SELECT_LINK_FXN, BATCH_SIZE, LEARNING_RATE, LIST_OF_NN_ELEMENTS, new_error_start, cost_fxn, batch_method,
                 gd_method, conv_method, lr_method, momentum_weight, rglztn_type, rglztn_fctr, conv_kill, pct_change,
                 conv_end_method, activation_constant, gd_iterations, non_neg_coeffs, allow_summary_print, summary_print_interval,
                 iteration):

        self.this_module = gmn.get_module_name(str(sys.modules[__name__]))
        fxn = inspect.stack()[0][3]

        self.DATA = DATA
        self.TARGET_VECTOR = TARGET_VECTOR
        self.data_run_orientation = data_run_orientation
        self.target_run_orientation = target_run_orientation
        self.ARRAY_OF_NODES = ARRAY_OF_NODES
        self.ARRAY_OF_NODES_BACKUP = deepcopy(self.ARRAY_OF_NODES)
        self.OUTPUT_VECTOR = OUTPUT_VECTOR
        self.SELECT_LINK_FXN = SELECT_LINK_FXN
        self.BATCH_SIZE = BATCH_SIZE
        self.LEARNING_RATE = LEARNING_RATE
        self.LIST_OF_NN_ELEMENTS = LIST_OF_NN_ELEMENTS
        self.new_error_start = new_error_start
        self.cost_fxn = cost_fxn.upper()
        self.batch_method = batch_method
        self.gd_method = gd_method
        self.conv_method = conv_method   # CONVERGENCE METHOD
        self.lr_method = lr_method

        self.momentum_weight = momentum_weight
        self.rglztn_type = rglztn_type
        self.rglztn_fctr = rglztn_fctr
        self.conv_kill = conv_kill
        self.pct_change = pct_change
        self.conv_end_method = conv_end_method
        self.activation_constant = activation_constant
        self.gd_iterations = gd_iterations
        self.non_neg_coeffs = non_neg_coeffs
        self.allow_summary_print = allow_summary_print
        self.summary_print_interval = summary_print_interval
        self.num_nodes = len(self.ARRAY_OF_NODES)

        self.data_rows, self.data_cols = gs.get_shape('DATA', self.DATA, self.data_run_orientation)

        if iteration != 0 and vui.validate_user_str(f'Reset iteration counter ({iteration}) to zero? (y/n) > ', 'YN') == 'Y':
            self.iteration = 0
        else:
            self.iteration = iteration

        self.LINK_FXNS = lf.define_links()

        self.OLD_UPDATE_ARRAYS = []
        for ARRAY in self.ARRAY_OF_NODES:
            self.OLD_UPDATE_ARRAYS.append(np.zeros(np.shape(ARRAY)))
            #  THE USED TO BE JUST A DEEPCOPY OF THE FIRST ARRAY_OF_GRADIENTS UNTIL 6-29-22, WHICH WORKED FINE

        self.OLD_RMS_ARRAYS = []
        for ARRAY in self.ARRAY_OF_NODES:
            self.OLD_RMS_ARRAYS.append(np.zeros(np.shape(ARRAY)))


    def divergence_handling(self):
        return 'T'
        # return vui.validate_user_str(f'Accept and end(a), ignore(i), adjust learning rate(l), divide all ' + \
        #                                 f'learning rates by 2(t)? > ', 'AILT')


    def early_stop_handling(self):
        return None


    def return_fxn(self):
        return self.ARRAY_OF_NODES, self.OUTPUT_VECTOR, self.iteration


    def run(self):

        fxn = inspect.stack()[0][3]

        # DECLARE INITIAL CONDITIONS IN THIS SCOPE
        self.new_error_start = float('inf')

        st.show_start_time('GD')
        start_time = datetime.now()  #.strftime("%m-%d-%Y %H:%M:%S")
        print('Running GD......')

        best_value = float('inf')  # USED TO FIND IF GD IS GYRATING AROUND MINIMUM
        no_improv_ctr = 0

        # SEED ARRAY_OF_GRADIENTS TO ESTABLISH INDICES
        ARRAY_OF_GRADIENTS = np.zeros(self.num_nodes, dtype=object)

        ARRAY_OF_z_x = np.zeros(self.num_nodes, dtype=object)
        ARRAY_OF_a_x_b4 = np.zeros(self.num_nodes, dtype=object)
        ARRAY_OF_a_x_after = np.zeros(self.num_nodes, dtype=object)

        DataClass = mlo.MLObject(
                                    self.DATA,
                                    self.data_run_orientation,
                                    name='DATA',
                                    return_orientation='COLUMN',
                                    return_format='AS_GIVEN',
                                    bypass_validation=True,
                                    calling_module=self.this_module,
                                    calling_fxn=fxn
        )

        TargetClass = mlo.MLObject(
                                    self.TARGET_VECTOR,
                                    self.target_run_orientation,
                                    name='TARGET',
                                    return_orientation='COLUMN',
                                    return_format='AS_GIVEN',
                                    bypass_validation=True,
                                    calling_module=self.this_module,
                                    calling_fxn=fxn
        )


        while self.iteration in range(self.gd_iterations):

            if (self.iteration + 1) % self.summary_print_interval == 0:
                print(f'\nRunning iteration {self.iteration + 1}......')

            # GENERATE RANDOM EXAMPLE BATCH*******************************************************************

            if self.batch_method in 'B':
                BATCH_TARGET_VECTOR = TargetClass.return_as_column()
                BATCH_DATA = DataClass.return_as_column()
            elif self.batch_method in 'MS':
                RANDOM_EXAMPLES = nnrc.new_np_random_choice(range(self.data_rows),
                                                        min(self.BATCH_SIZE[self.iteration], self.data_rows), replace=False)

                BATCH_TARGET_VECTOR = TargetClass.return_rows(RANDOM_EXAMPLES, return_orientation='COLUMN', return_format='ARRAY')

                BATCH_DATA = DataClass.return_rows(RANDOM_EXAMPLES, return_orientation='COLUMN', return_format='AS_GIVEN')

                del RANDOM_EXAMPLES


            # END GENERATE EXAMPLE BATCH***************************************************************

            # GD ENGINES******************************************************************************
            # ******************************************************************************************
            # COORDINATE METHOD**************************************************************************************
            if self.gd_method == 'C':
                # NOTES 4-30-22 "BATCH _OUTPUT_VECTOR, _TARGET_VECTOR, _MATRIX", FOR CALC PURPOSES, WHETHER FULL OR MINI-BATCH
                # self.OUTPUT_VECTOR, self.new_error_start IS OUTPUT OF FULL BATCH, FOR REPORTING/DISPLAYING.
                if (self.iteration + 1) % self.summary_print_interval == 0:
                    print(f'CD ITERATION {self.iteration + 1} ERROR START: {self.new_error_start}')

                self.ARRAY_OF_NODES, self.OUTPUT_VECTOR, self.new_error_start = \
                    ce.cd_engine('Min', self.ARRAY_OF_NODES, self.LIST_OF_NN_ELEMENTS, self.SELECT_LINK_FXN, BATCH_TARGET_VECTOR,
                        BATCH_DATA, [], self.LEARNING_RATE, self.cost_fxn, self.non_neg_coeffs,
                        self.new_error_start, self.rglztn_type, self.rglztn_fctr, self.activation_constant, self.iteration)

                if (self.iteration + 1) % self.summary_print_interval == 0:
                    print(f'CD ITERATION {self.iteration + 1} ERROR END: {self.new_error_start}')

            # END COORDINATE METHOD*************************************************************************************

            # GRADIENT METHODS (GD, RMS, ADAM, N) *************************************************************************************
            elif self.gd_method in 'G':
                # NOTES 4-30-22 "BATCH _OUTPUT_VECTOR, _TARGET_VECTOR, _MATRIX", FOR CALC PURPOSES, WHETHER FULL OR MINI-BATCH
                # self.OUTPUT_VECTOR, self.new_error_start IS OUTPUT OF FULL BATCH, FOR REPORTING/DISPLAYING.
                # self.TARGET_VECTOR UNCHOPPED ORIGINAL

                if (self.iteration + 1) % self.summary_print_interval == 0:
                    print(f"\nITERATION {self.iteration + 1} TOTAL ERROR START".ljust(
                        len(f'ITERATION {self.iteration + 1} TOTAL ERROR AFTER UPDATE TO NODE X ')) + \
                            f" = {self.new_error_start}")  # MUMBO JUMBO TO GET PRINTOUTS TO LINE UP

                # USING ARRAY OF NODES FROM LAST ITERATION, SEE WHAT new_error_start IS FOR THE NEW BATCH_DATA, USE
                # THIS FOR DIVERGENCE HANDLING
                if self.batch_method != 'B' and self.BATCH_SIZE[self.iteration] < self.data_rows:
                    DUMMY_OUTPUT_VECTOR = ovc.output_vector_calc(BATCH_DATA, self.ARRAY_OF_NODES, self.SELECT_LINK_FXN, [],
                                                                 self.activation_constant)
                    self.new_error_start = ec.error_calc(self.ARRAY_OF_NODES, BATCH_TARGET_VECTOR, DUMMY_OUTPUT_VECTOR,
                             self.cost_fxn, self.new_error_start, self.SELECT_LINK_FXN, self.rglztn_type, self.rglztn_fctr)
                batch_divergence_start = self.new_error_start

                # FORWARD PROP ########################################################################################
                # MMULT BATCH_DATA EXAMPLES INTO NN

                BATCH_OUTPUT_VECTOR, ARRAY_OF_z_x, ARRAY_OF_a_x_b4, ARRAY_OF_a_x_after = \
                    ovc.output_vector_calc_backprop(BATCH_DATA, self.ARRAY_OF_NODES, ARRAY_OF_z_x, ARRAY_OF_a_x_b4,
                                    ARRAY_OF_a_x_after, self.SELECT_LINK_FXN, [], self.activation_constant)

                if self.SELECT_LINK_FXN[-1].upper() != 'SOFTMAX':  # THIS IS ALSO FOR MULTI-OUT
                    # GENERATE h_x WHEN / IF NEEDED LATER FOR HESSIAN
                    h_x = BATCH_OUTPUT_VECTOR  # LEAVE THIS LOWERCASE SO NOT TO CONFUSE WITH HESSIAN
                    SMAX_h_x = []  # DUMMY FOR INGESTION INTO generate_hessian()

                elif self.SELECT_LINK_FXN[-1].upper() == 'SOFTMAX':
                    # GENERATE SMAX_h_x WHEN / IF NEEDED LATER FOR HESSIAN
                    SMAX_h_x = BATCH_OUTPUT_VECTOR

                    # GENERATE h_x(y=1) WHEN / IF NEEDED LATER FOR HESSIAN
                    h_x = [[]]
                    for label_idx in range(len(BATCH_TARGET_VECTOR)):
                        for example_idx in range(len(BATCH_TARGET_VECTOR[label_idx])):
                            if BATCH_TARGET_VECTOR[label_idx][example_idx] == 1:
                                h_x[0].append(BATCH_OUTPUT_VECTOR[label_idx][example_idx])
                # END FORWARD PROP #################################################################################################

                # #########################################################################################################
                # ###################### BUILD GRADIENT VECTOR(S) ######################################################

                # BACK PROP ###############################################################################################
                # BUILD cost_fxn DERIVATIVE WRT OUTPUT

                if self.cost_fxn == 'S':
                    dC_dan = NS().subtractf(BATCH_OUTPUT_VECTOR, BATCH_TARGET_VECTOR)

                elif self.cost_fxn == 'L' and len(BATCH_TARGET_VECTOR) == 1:  # FOR REGULAR TARGET
                    dC_dan = NS().dividef(
                        NS().subtractf(BATCH_OUTPUT_VECTOR, BATCH_TARGET_VECTOR),
                        NS().multiplyf(BATCH_OUTPUT_VECTOR, NS().subtractf(1, BATCH_OUTPUT_VECTOR))
                    )

                elif self.cost_fxn == 'L' and len(BATCH_TARGET_VECTOR) > 1:  # FOR SOFTMAX TARGET

                    GRAD_ARRAY = []
                    for category_idx in range(len(BATCH_TARGET_VECTOR)):
                        GRAD_ARRAY.append([])
                        for example_idx in range(len(BATCH_TARGET_VECTOR[category_idx])):
                            if BATCH_TARGET_VECTOR[category_idx][example_idx] == 1:
                                GRAD_ARRAY[-1].append(-1 / BATCH_OUTPUT_VECTOR[category_idx][example_idx])
                            elif BATCH_TARGET_VECTOR[category_idx][example_idx] == 0:
                                GRAD_ARRAY[-1].append(1 / (1 - BATCH_OUTPUT_VECTOR[category_idx][example_idx]))
                            else:
                                raise Exception(f'THERE IS A NON-BINARY IN TARGET VECTOR IN {__name__} FOR SOFTMAX')

                    dC_dan = GRAD_ARRAY

                elif self.cost_fxn == 'U':
                    # NOT DONE
                    dC_dan = NS().subtractf(1, 2 * BATCH_OUTPUT_VECTOR)

                # BUILD OUTPUT DERIVATIVE WRT z_n
                dan_dzn = lf.link_derivative(BATCH_OUTPUT_VECTOR, self.SELECT_LINK_FXN[-1])

                # DOT dC_dan WITH dan_dzn
                DOTTED_FINAL_DERIVATIVES = NS().multiplyf(dC_dan, dan_dzn)

                for target_node_idx in range(self.num_nodes - 1, -1, -1):
                    # CALCULATE THE ODDBALL CASE OF THE LAST GRADIENT ONCE, THEN CARRY THOUGH THE LOOP FOR REMAINING GRADS
                    if target_node_idx == self.num_nodes - 1:

                        ARRAY_OF_GRADIENTS[target_node_idx] = np.matmul(DOTTED_FINAL_DERIVATIVES,
                                                                        np.transpose(ARRAY_OF_a_x_after[-1])
                                                                        )
                        STUB = DOTTED_FINAL_DERIVATIVES

                    else:
                        if self.activation_constant == 1:
                            INTERMEDIATE2 = np.array([X[:-1] for X in self.ARRAY_OF_NODES[target_node_idx + 1]])
                        else:
                            INTERMEDIATE2 = self.ARRAY_OF_NODES[target_node_idx + 1]

                        INTERMEDIATE2 = np.matmul(np.transpose(INTERMEDIATE2), STUB)
                        INTERMEDIATE2 = NS().multiplyf(
                            lf.link_derivative(ARRAY_OF_a_x_b4[target_node_idx], self.SELECT_LINK_FXN[target_node_idx]),
                            INTERMEDIATE2)
                        STUB = INTERMEDIATE2
                        ARRAY_OF_GRADIENTS[target_node_idx] = np.matmul(INTERMEDIATE2, np.transpose(
                            ARRAY_OF_a_x_after[target_node_idx]))

                # END BACK PROP ###############################################################################################

                # ADJUST BACKPROP GRADIENTS WITH REGULARIZATION #########################################################
                for node_idx in range(len(ARRAY_OF_GRADIENTS)):

                    if self.rglztn_type == 'L1':

                        '''UPDATED TO ∂J/∂W + λ/2*sign(W)'''

                        ARRAY_OF_GRADIENTS[node_idx] = \
                            NS().addf(ARRAY_OF_GRADIENTS[node_idx],
                                          NS().multiplyf(0.5 * self.rglztn_fctr,
                                                         np.sign(self.ARRAY_OF_NODES[node_idx])
                                                         )
                                          )

                    elif self.rglztn_type == 'L2':

                        ''' MODIFIED FROM TO CURRENT FORM TO REMOVE ALPHA
                            W = W − α*∂J/∂W − α*λ/2*∂(WTW)/∂W
                              = (1 − αλ)W − α*∂J/∂W'''

                        '''UPDATED TO ∂J/∂W + λW'''

                        ARRAY_OF_GRADIENTS[node_idx] = \
                            NS().addf(
                                NS().multiplyf(self.rglztn_fctr, self.ARRAY_OF_NODES[node_idx]),
                                                ARRAY_OF_GRADIENTS[node_idx]
                            )

                # END ADJUST BACKPROP GRADIENTS WITH REGULARIZATION #########################################################

                # ###################### END BUILD GRADIENT VECTOR(S) ##################################################
                #########################################################################################################

                # CONVENTIONAL DESCENT METHODS (GD, RMS, ADAM) **************************************************************************
                if self.conv_method in 'GRA':

                    # UPDATE PARAMETERS ################################################################################
                    layer_divergence_error_holder, DIVERGENCE_LAYERS = self.new_error_start, []
                    for node_idx in range(len(ARRAY_OF_GRADIENTS)):
                        # BUILD UPDATE MATRICES ########################################################################
                        if self.conv_method == 'G':   # REGULAR GD W/ MOMENTUM
                            ARRAY_OF_GRADIENTS[node_idx] = gdgd.gd_gd(
                                ARRAY_OF_GRADIENTS[node_idx], self.OLD_UPDATE_ARRAYS[node_idx], self.momentum_weight)
                            self.OLD_UPDATE_ARRAYS[node_idx] = deepcopy(ARRAY_OF_GRADIENTS[node_idx])

                        elif self.conv_method == 'R':   # RMS
                            ARRAY_OF_GRADIENTS[node_idx], self.OLD_RMS_ARRAYS[node_idx] = \
                                gdr.gd_rmsprop(ARRAY_OF_GRADIENTS[node_idx], self.OLD_RMS_ARRAYS[node_idx])

                        elif self.conv_method == 'A':    # ADAM
                            ARRAY_OF_GRADIENTS[node_idx], self.OLD_UPDATE_ARRAYS[node_idx], self.OLD_RMS_ARRAYS[node_idx] = \
                                gda.gd_adam(ARRAY_OF_GRADIENTS[node_idx], self.OLD_UPDATE_ARRAYS[node_idx], self.OLD_RMS_ARRAYS[node_idx],
                                            self.momentum_weight, self.iteration+1)

                        # END BUILD UPDATE MATRICES #####################################################################

                        # APPLY GRADIENT UPDATE ##########################################################################

                        self.ARRAY_OF_NODES[node_idx] = self.ARRAY_OF_NODES[node_idx] - \
                                        self.LEARNING_RATE[node_idx][self.iteration] * ARRAY_OF_GRADIENTS[node_idx]

                        # END APPLY GRADIENT UPDATE ##########################################################################

                        if self.non_neg_coeffs == 'Y':
                            self.ARRAY_OF_NODES[node_idx] = \
                                lf.link_fxns(self.ARRAY_OF_NODES[node_idx], 'ReLU_lower')

                        # THIS IS UNDER AON UPDATE LOOP TO SHOW CHANGE IN ERROR FOR EACH NODE UPDATE DURING BACKPROP
                        # USE THE BATCH OBJECTS, NOT THE FULL OBJECTS!
                        self.OUTPUT_VECTOR = ovc.output_vector_calc(BATCH_DATA, self.ARRAY_OF_NODES, self.SELECT_LINK_FXN,
                                                               [], self.activation_constant)

                        # 10-1-22 self.new_error_start WAS '', GIVING can only concatenate str (not "numpy.float64") to str
                        self.new_error_start = ec.error_calc(self.ARRAY_OF_NODES, BATCH_TARGET_VECTOR, self.OUTPUT_VECTOR,
                                self.cost_fxn, '', self.SELECT_LINK_FXN, self.rglztn_type, self.rglztn_fctr)

                        if (self.iteration + 1) % self.summary_print_interval == 0:
                            # USE FULL OBJECTS HERE FOR OUTPUT & ERROR... JUST FOR DISPLAY TO SCREEN
                            if self.batch_method == 'B' or self.BATCH_SIZE[self.iteration] > self.data_rows:
                                print( f'ITERATION {self.iteration + 1} TOTAL ERROR AFTER UPDATE TO NODE {node_idx} = {self.new_error_start}')
                            else:
                                self.OUTPUT_VECTOR = ovc.output_vector_calc(DataClass.OBJECT, self.ARRAY_OF_NODES,
                                    self.SELECT_LINK_FXN, [], self.activation_constant)

                                self.new_error_start = ec.error_calc(self.ARRAY_OF_NODES, TargetClass.OBJECT,
                                     self.OUTPUT_VECTOR, self.cost_fxn, '', self.SELECT_LINK_FXN, self.rglztn_type, self.rglztn_fctr)

                                print(f'ITERATION {self.iteration + 1} TOTAL ERROR AFTER UPDATE TO NODE {node_idx} = {self.new_error_start}')

                        # DIVERGENCE HANDLING ###########################################################################
                        if self.new_error_start > layer_divergence_error_holder:
                            DIVERGENCE_LAYERS.append(str(node_idx))

                        layer_divergence_error_holder = self.new_error_start

                        if node_idx == self.num_nodes-1:
                            batch_divergence_end = self.new_error_start
                        # END DIVERGENCE HANDLING ###########################################################################

                    # END UPDATE PARAMETERS ############################################################################

                # END CONVENTIONAL DESCENT METHODS (GD, RMS, ADAM) **************************************************************************

                # APPLY GRADIENT IN NEWTONS METHOD ####################################################################
                elif self.conv_method == 'N':
                    # HAVE TO GENERATE THIS CRAZY MESS BECAUSE SOFTMAX REQUIRES A DIFFERENT HESSIAN FOR EACH LABEL

                    busted_hessian = False  # BUSTED HESSIAN BREAK INITIAL SETTING

                    H = gh.generate_hessian(BATCH_DATA, BATCH_TARGET_VECTOR, h_x, SMAX_h_x, self.SELECT_LINK_FXN[0])

                    self.LEARNING_RATE = np.ones((self.num_nodes, self.gd_iterations), dtype=np.int32)  # ONLY IF NEWTONS METHOD

                    for l_h_idx in range(len(H)):  # l_h = LABEL_HESSIAN
                        try:
                            if isinstance(H, dict): H_inv = np.linalg.inv(sd.unzip_to_ndarray_float64(H)[0])
                            elif isinstance(H, (np.ndarray, list, tuple)): H_inv = np.linalg.inv(np.array(H, dtype=np.float64))
                            else: DUM = input(f'\n*** INVALID DATA FORM FOR H, HIT ENTER TO CONTINUE ***\n')
                            print(f'\nAON[{l_h_idx}] HESSIAN invert OK!!!\n')
                        except:
                            # IF H DIDNT INVERT, SEND USER DOWN TO CONFIG
                            busted_hessian = True
                            print(f'H #{l_h_idx + 1} wont invert, change to regular gradient descent.')
                            break

                        ARRAY_OF_GRADIENTS[l_h_idx] = np.matmul(H_inv, ARRAY_OF_GRADIENTS[l_h_idx], dtype=float)

                        self.ARRAY_OF_NODES[l_h_idx] += self.LEARNING_RATE[l_h_idx][self.iteration] * ARRAY_OF_GRADIENTS[l_h_idx]

                        if self.non_neg_coeffs == 'Y':
                            self.ARRAY_OF_NODES[0] = lf.link_fxns(self.ARRAY_OF_NODES[0], 'ReLU_lower')

                        self.OUTPUT_VECTOR = ovc.output_vector_calc(
                            DataClass.OBJECT, self.ARRAY_OF_NODES, self.SELECT_LINK_FXN, [], self.activation_constant)

                        self.new_error_start = \
                            ec.error_calc(self.ARRAY_OF_NODES, TargetClass.OBJECT, self.OUTPUT_VECTOR, self.cost_fxn,
                                            '', self.SELECT_LINK_FXN, self.rglztn_type, self.rglztn_fctr)

                        if (self.iteration + 1) % self.summary_print_interval == 0:
                            print(f'ITERATION {self.iteration + 1} TOTAL ERROR AFTER UPDATE TO NODE {node_idx} = {self.new_error_start}\n')

                # END APPLY GRADIENT IN NEWTONS METHOD ####################################################################

            # END GRADIENT METHODS (GD, RMS, ADAM, N) **************************************************************************************
            # ******************************************************************************************************

            # AFTER FINISHING BACKPROP FOR CURRENT ITERATION, CALCULATE OUTPUT & ERROR FOR FULL OBJECTS
            if self.batch_method != 'B' and self.BATCH_SIZE[self.iteration] < self.data_rows:
                self.OUTPUT_VECTOR = ovc.output_vector_calc(DataClass.OBJECT, self.ARRAY_OF_NODES, self.SELECT_LINK_FXN,
                                         [], self.activation_constant)

                self.new_error_start = ec.error_calc(self.ARRAY_OF_NODES, TargetClass.OBJECT, self.OUTPUT_VECTOR,
                                         self.cost_fxn, '', self.SELECT_LINK_FXN, self.rglztn_type, self.rglztn_fctr)

            # STUFF FOR MANAGING DIVERGENCE
            # CANNOT USE new_error_start AFTER THE LAST ITERATION) BECAUSE A NEW MINI-BATCH
            # (NOT FULL BATCH) IRREGARDLESS OF LEARNING RATES MAY HAVE SHOT TO A NEW POINT WHOSE STARTING ERROR IS WORSE
            # THAN THE LAST (IN WHICH CASE, IF WE WERE TO CALL "DIVERGENCE" BASED ON COMPARISON TO THE LAST ITERATIONS
            # ERROR, WE WOULD BE PUNISHING THE LEARNING RATES FOR SOMETHING THAT WASNT THEIR FAULT.  SO CALCULATE
            # new_error_start FOR THE NEW LANDING POINT B4 APPLYING BACKPROP UPDATES, CALL THAT new_error_start.  THIS
            # new_error_start IS batch_divergence_start (bds).  BACKPROP THRU ALL THE LAYERS, EACH TIME MEASURING IF THE
            # LAYER UPDATE MADE THE ERROR WORSE THAN BEFORE, WHICH MEANS THAT THE LEARNING RATE FOR THAT LAYER IS TOO HIGH.
            # THIS IS layer_divergence_error_holder. AFTER BACK-PROPPING THRU ALL THE LAYERS, batch_divergence_end (bde)
            # IS new_error_start.  IF bde > bds, THEN THE LEARNING RATES WERE BAD, WHICH MEANS AT LEAST ONE THING MUST BE
            # IN DIVERGENCE_LAYERS

            # DIVERGENCE CAUSED BY LEARNING RATES IF
            # --- BATCH SIZE METHOD IS BATCH OR [ANY BATCH METHOD AND BATCH SIZE FOR CURRENT iteration IS >= FULL BATCH SIZE]
            #       (GRADIENT CANNOT BE WRONG)
            # AND
            # --- batch_divergence_end > batch_divergence_start (END ERROR THIS iter > START ERROR THIS iter)
            # OR
            # --- ANY BATCH METHOD AND ERROR INCREASES AT ANY POINT DURING BACKPROP (layer_divergence_error_holder AFTER
            #       UPDATING THE CURRENT LAYER IS > layer_divergence_error_holder AFTER UPDATING THE LAST LAYER.
            #       RECORD THE VIOLATING LAYER IN "DIVERGENCE_LAYERS".

            # DONT WORRY ABOUT DIVERENCE IF ON LAST ITER
            if self.iteration + 1 < self.gd_iterations and self.conv_method in 'GRA' and \
                batch_divergence_end > batch_divergence_start:
                if len(DIVERGENCE_LAYERS) == 0:
                    raise Exception(f'\n*** DIVERGENCE DURING BACKPROP BUT DIVERGENCE_LAYERS IS EMPTY.***\n')

                print(f'\nLAYER(S) {", ".join(DIVERGENCE_LAYERS)} DIVERGED ON ITERATION {self.iteration} of {self.gd_iterations}, '
                      f'DIVIDING THOSE LEARNING RATES BY 2.')

                __ = self.divergence_handling()

                if __ == 'A': break
                elif __ == 'I': pass
                elif __ == 'L':
                    for node in range(len(self.LEARNING_RATE)):
                        print(f'NODE {node}) {[[f"{_:.5g}" for _ in __] for __ in self.LEARNING_RATE][node][:10]}')
                    self.LEARNING_RATE = lrs.SetLearningRate(self.LEARNING_RATE, self.ARRAY_OF_NODES,
                            self.SELECT_LINK_FXN, self.lr_method, self.gd_iterations).learning_rate_run()
                elif __ == 'T':
                    for div_node_idx in DIVERGENCE_LAYERS:
                        div_node_idx = int(div_node_idx)
                        self.LEARNING_RATE[div_node_idx] = np.divide(self.LEARNING_RATE[div_node_idx], 2)
                        print( f'NEW LEARNING_RATE[{div_node_idx}][{self.iteration + 1}] = {self.LEARNING_RATE[div_node_idx][self.iteration + 1]}')

                    if False not in [self.LEARNING_RATE[_][self.iteration + 1] <= 1e-6 for _ in range(len(self.LEARNING_RATE))]:
                        print(f'\nAll learning rates have fallen below 1e-6, end current learning cycle.'.upper())
                        break

            if self.gd_method == 'C':
                if busted_hessian:
                    print('\nBusted Hessian.\n')
                    break

            if self.allow_summary_print == 'Y':
                if (self.iteration + 1) % self.summary_print_interval == 0:
                    nnss.NNSummaryStatisticsPrint(self.OUTPUT_VECTOR, TargetClass.OBJECT, self.SELECT_LINK_FXN[-1],
                        self.new_error_start, self.ARRAY_OF_NODES, self.iteration + 1, self.gd_iterations).print()

                # 4-16-22 TRY PAUSE HERE TO SEE IF HELPS PREVENT SLOWDOWN AFTER LONG RUN
                time.sleep(1)

            if self.iteration == self.gd_iterations - 1:
                print(f'\nGradient descent accomplished {self.iteration + 1} of {self.gd_iterations} user-specified iterations')
                nnss.NNSummaryStatisticsPrint(self.OUTPUT_VECTOR, TargetClass.OBJECT, self.SELECT_LINK_FXN[-1],
                            self.new_error_start, self.ARRAY_OF_NODES, self.iteration + 1, self.gd_iterations).print()
                st.show_end_time('GD')
                wls.winlinsound(888, 100)
                break

            # STUFF FOR MANAGING CONVERGENCE KILL
            no_improv_ctr, self.conv_kill, best_value, abort = ni.NoImprov(self.new_error_start, self.iteration + 1, 1,
                                    no_improv_ctr, self.conv_kill, best_value, self.pct_change, 'NN',
                                    conv_end_method=self.conv_end_method).min()

            if abort:
                print(f'\nTOTAL ERROR has not achieved {str(self.pct_change)}% change for {no_improv_ctr} iteration(s) after ' + \
                    f'{self.iteration + 1} of {self.gd_iterations} iterations. Automatically ending gradient descent.')

                print('\nStart time: ', start_time.strftime("%m-%d-%Y %H:%M:%S"))
                st.show_end_time('GD')
                end_time = datetime.now()  # .strftime("%m-%d-%Y %H:%M:%S")
                try: print(f'Elapsed time: {round((end_time - start_time).seconds / 60, 2)} minutes')
                except: print('Subtracting end and start times is giving error')

                break

            if no_improv_ctr == self.conv_kill and self.iteration + 1 != self.gd_iterations:
                print(f'\nTOTAL ERROR has not achieved {str(self.pct_change)}% change for {no_improv_ctr} iteration(s) after ' + \
                      f'{self.iteration + 1} of {self.gd_iterations} iterations')
                nnss.NNSummaryStatisticsPrint(self.OUTPUT_VECTOR, TargetClass.OBJECT, self.SELECT_LINK_FXN[-1],
                             self.new_error_start, self.ARRAY_OF_NODES, self.iteration + 1, self.gd_iterations).print()

                if os.name == 'nt': wls.winlinsound(888, 500)
                no_improv_ctr = 0  # RESET TO ZERO FOR BOTH IGNORE AND BREAK

                if vui.validate_user_str('Ignore (i) or break (b)? > ', 'IB') == 'I':
                    continue
                else:
                    print('\nStart time: ', start_time.strftime("%m-%d-%Y %H:%M:%S"))
                    st.show_end_time('GD')
                    end_time = datetime.now()  # .strftime("%m-%d-%Y %H:%M:%S")
                    try: print(f'Elapsed time: {round((end_time - start_time).seconds / 60, 2)} minutes')
                    except: print('Subtracting end and start times is giving error')

                    break

            if self.early_stop_handling() == 'BREAK':
                # RESTORE SNAPSHOT OF AON FOR LOWEST DEV ERROR
                self.ARRAY_OF_NODES = deepcopy(self.ARRAY_OF_NODES_BACKUP)
                # UPDATE OUTPUT_VECTOR W RESTORED ARRAY_OF_NODES BEFORE return
                self.OUTPUT_VECTOR = ovc.output_vector_calc(
                    DataClass.OBJECT, self.ARRAY_OF_NODES, self.SELECT_LINK_FXN, [], self.activation_constant)
                print(f'\n*** Training cycle stopped early on iteration {self.iteration+1} of {self.gd_iterations} for '
                      f'divergence of validation error. ***\n')
                break

            self.iteration += 1

        del TargetClass, DataClass

        return self.return_fxn()



class NNCoreRunCode_MLPackage(NNCoreRunCode):
    def __init__(self, DATA, TARGET_VECTOR, DEV_DATA, DEV_TARGET, data_run_orientation, target_run_orientation, ARRAY_OF_NODES,
                 OUTPUT_VECTOR, SELECT_LINK_FXN, BATCH_SIZE, LEARNING_RATE, LIST_OF_NN_ELEMENTS, new_error_start, cost_fxn,
                 batch_method, gd_method, conv_method, lr_method, momentum_weight, rglztn_type, rglztn_fctr, conv_kill,
                 pct_change, conv_end_method, activation_constant, gd_iterations, non_neg_coeffs, allow_summary_print,
                 summary_print_interval, early_stop_interval, iteration):

        super().__init__(DATA, TARGET_VECTOR, data_run_orientation, target_run_orientation, ARRAY_OF_NODES, OUTPUT_VECTOR,
                 SELECT_LINK_FXN, BATCH_SIZE, LEARNING_RATE, LIST_OF_NN_ELEMENTS, new_error_start, cost_fxn, batch_method,
                 gd_method, conv_method, lr_method, momentum_weight, rglztn_type, rglztn_fctr, conv_kill, pct_change,
                 conv_end_method, activation_constant, gd_iterations, non_neg_coeffs, allow_summary_print,
                 summary_print_interval, 0)

        self.DEV_DATA = DEV_DATA
        self.DEV_TARGET = DEV_TARGET
        self.early_stop_interval = early_stop_interval
        self.iteration = 0
        self.early_stop_error_holder = float('inf')


    def divergence_handling(self):
        # WHEN DIVERGENCE, SIMPLY DIVIDE ALL LEARNING RATES BY TWO FOR K-FOLD, FOR CONTINUITY
        return 'T'


    def early_stop_handling(self):
        if ((self.iteration + 1) % self.early_stop_interval == 0 or self.iteration == 0) and self.early_stop_interval != 1e12:
            # 1e12 INDICATED early_stop IS OFF
            DEV_OUTPUT = ovc.output_vector_calc(self.DEV_DATA, self.ARRAY_OF_NODES, self.SELECT_LINK_FXN, [],
                                                self.activation_constant)

            dev_error = ec.error_calc(self.ARRAY_OF_NODES, self.DEV_TARGET, DEV_OUTPUT, self.cost_fxn, 0,
                                      self.SELECT_LINK_FXN, self.rglztn_type, 0)

            print(f'\nITERATION {self.iteration + 1} VALIDATION ERROR = {dev_error}\n')

            if dev_error >= self.early_stop_error_holder:
                return 'BREAK'
            else:
                self.early_stop_error_holder = dev_error
                self.ARRAY_OF_NODES_BACKUP = deepcopy(self.ARRAY_OF_NODES)  # KEEP A SNAPSHOT OF AON FOR LAST LOWEST DEV ERROR








if __name__ == '__main__':
    import pandas as pd
    from read_write_file.generate_full_filename import base_path_select as bps

    # num_rows = 10
    # num_cols = 3
    # DATA = np._random_.randn(num_cols,num_rows)
    # TARGET_VECTOR = np._random_.randint(0,2,(1,num_rows))

    basepath = bps.base_path_select()
    DF = pd.DataFrame(pd.read_csv(basepath + r'rGDP.csv', nrows=200))
    # TARGET_VECTOR = np.array([np.sign(DF['rGDP'].to_numpy())])

    _len = len(DF)
    train_split = int(round(0.7 * _len, 0))

    # DATA = DF.drop(columns=['Date','Rec','rGDP','OIL'])#.to_numpy()
    ALL_DATA = DF[['POP', 'U3SA', 'U6SA']]
    DATA = ALL_DATA.iloc[:train_split,:].to_numpy().transpose()
    DEV_DATA = ALL_DATA.iloc[train_split:, :].to_numpy().transpose()
    data_run_orientation = 'COLUMN'

    RAW_TARGET_VECTOR = DF['rGDP'].to_numpy()
    TARGET_VECTOR = RAW_TARGET_VECTOR.transpose()[:train_split]
    TARGET_VECTOR.shape = (1,train_split)
    DEV_TARGET = RAW_TARGET_VECTOR.transpose()[train_split:]
    DEV_TARGET.shape = (1,len(DF)-train_split)
    target_run_orientation = 'COLUMN'

    # print(DATA)
    # print()
    # scaler = StandardScaler()
    # scaler.fit(DATA)
    # DATA = np.array(scaler.transform(DATA)).transpose()

    DATA = sd.zip_list(DATA)

    from ML_PACKAGE.MLREGRESSION import MLRegression as mlr

    NAMES = ['self.XTX_determinant',
    'self.COEFFS',
    'self.PREDICTED',
    'self.P_VALUES',
    'self.r',
    'self.R2',
    'self.R2_adj',
    'self.F']
    # 4/15/23 BEAR FIX THIS ARGS/KWARGS, MLRegression CHANGED
    BLEH = mlr.MLRegression(DATA=DATA, data_given_orientation=data_run_orientation,
                            DATA_TRANSPOSE=None, TARGET=TARGET_VECTOR, target_given_orientation=target_run_orientation,
                            TARGET_TRANSPOSE=None,
                            TARGET_AS_LIST=None, has_intercept=False,intercept_math=True,
                            safe_matmul=True).run()
    print()
    [print(f'{NAMES[_]} ) {BLEH[_]}') for _ in range(len(NAMES))]
    print()


    if isinstance(DATA, (list, tuple, np.ndarray)): num_cols, num_rows = len(DATA), len(DATA[0])
    elif isinstance(DATA, dict): num_cols, num_rows = sd.outer_len(DATA), sd.inner_len(DATA)
    activation_constant = 1
    ARRAY_OF_NODES = np.array([np.random.randn(3,num_cols), np.random.randn(1,3+activation_constant)], dtype=object)
    # ARRAY_OF_NODES = np.array([np._random_.randn(1, 3)], dtype=object)
    OUTPUT_VECTOR = [0 for _ in range(num_rows)]


    SELECT_LINK_FXN = ['Logistic', 'None']
    gd_iterations = 1000
    LEARNING_RATE = [[1+.01*_ for _ in range(gd_iterations)], [1+.01*_ for _ in range(gd_iterations)]]
    LIST_OF_NN_ELEMENTS = []
    for node_idx in range(len(ARRAY_OF_NODES)):
        for neuron_idx in range(len(ARRAY_OF_NODES[node_idx])):
            for param_idx in range(len(ARRAY_OF_NODES[node_idx][neuron_idx])):
                LIST_OF_NN_ELEMENTS.append([node_idx, neuron_idx, param_idx])
    new_error_start = 0
    cost_fxn = 'S'
    batch_method = 'M'
    if batch_method == 'B':
        BATCH_SIZE = [num_rows for _ in range(gd_iterations)]
    elif batch_method == 'M':
        BATCH_SIZE = [min(2*_+2, num_rows) for _ in range(gd_iterations)]

    gd_method = 'G'
    conv_method = 'A'
    lr_method = 'S'
    momentum_weight = 0.5
    rglztn_type = 'L2'
    rglztn_fctr = 1
    conv_kill = 100
    pct_change = 0.1
    conv_end_method = 'KILL'
    non_neg_coeffs = 'N'
    allow_summary_print = 'Y'
    summary_print_interval = 1
    early_stop_interval = 1
    iteration = 0

    # from debug import IdentifyObjectAndPrint as ioap
    # OBJS = [DATA, TARGET_VECTOR, ARRAY_OF_NODES[0], ARRAY_OF_NODES[1]]
    # NAMES = ['DATA', 'TARGET', 'ARRAY_OF_NODES[0]', 'ARRAY_OF_NODES[1]']
    # HEADERS = [[str(_) for _ in range(num_cols)], ['TARGET'], [f'AON[0][{_}]' for _ in range(len(ARRAY_OF_NODES[0]))], [f'AON[1][{_}]' for _ in range(len(ARRAY_OF_NODES[1]))]]
    # for obj_idx in range(len(OBJS)):
    #     print(f'\nPRINTING {NAMES[obj_idx]}')
    #     ioap.IdentifyObjectAndPrint(OBJS[obj_idx], NAMES[obj_idx], __name__, 10, 5).run_print_as_df(
    #         df_columns=HEADERS[obj_idx], orientation='column')


    ARRAY_OF_NODES, OUTPUT_VECTOR, iteration = \
    NNCoreRunCode_MLPackage(DATA, TARGET_VECTOR, DEV_DATA, DEV_TARGET, data_run_orientation, target_run_orientation,
                 ARRAY_OF_NODES, OUTPUT_VECTOR, SELECT_LINK_FXN, BATCH_SIZE, LEARNING_RATE, LIST_OF_NN_ELEMENTS,
                 new_error_start, cost_fxn, batch_method, gd_method, conv_method, lr_method, momentum_weight, rglztn_type,
                 rglztn_fctr, conv_kill, pct_change, conv_end_method, activation_constant, gd_iterations, non_neg_coeffs,
                 allow_summary_print, summary_print_interval, early_stop_interval, iteration).run()












