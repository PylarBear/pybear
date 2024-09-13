import numpy as n
from datetime import datetime
from copy import deepcopy
import sparse_dict as sd
from data_validation import validate_user_input as vui
from ML_PACKAGE.GENERIC_PRINT import show_time as st
from general_data_ops import NoImprov as ni
from ML_PACKAGE.SVM_PACKAGE import svm_output_calc as svmoc, svm_error_calc as sec

'''
return_fxn               Return from run()
use_early_stopping       Indicates whether early stopping for dev/_validation error.
early_stop_handling      Calculation of dev _validation errar for early stop.
alpha_handling           Allows (or disallows) user to choose to reset alphas or keep old values.
objective_function       Obsoleted. Calculates the Lagrangian objective, that is maximized when optimal alphas are found. 
non_bound_idxs           Indices of examples that are not "in the river" or "on the river bank."  alpha should be zero.
bound_idxs               Indices of examples that are "in the river" or "on the river bank."  alpha should be >0 and <C
KKT_test                 Test for violation of KKT rules for out-of-bounds, on the river bank, or in the river examples.
w                        Obsoleted. Calculation of the boundary equation.  Could only be used for linear kernel.
F                        Calculate function value for all examples.  Function value is not functional margin!
f                        Calculate function value for a single example.  Function value is not functional margin!
E                        Calculate error for all examples.
e                        Calculate individual example error for Platt error update rules.
eta                      Second derivative of the objective function, usually (hopefully) less than zero.
L_H_bound_check          Calculate the bounds L and H for a2 (lies between 0 and C)
joint_alpha_optimizer    Core SMO alpha optimizer algorithm.
build_boundary_objects   Build self.SUPPORT_VECTORS, self.SUPPORT_ALPHAS, self.SUPPORT_TARGETS, self.SUPPORT_KERNELS 
run                      SVMCoreSMOCode exe module.
'''


class SVMCoreSMOCode:
    # MAXIMIZING OBJECTIVE FUNCTION
    # W(a) = sum(all alphas) - 0.5 * sum over i, sum over j [ (yi * yj * ai * aj * k<xi, xj> ]
    # SUBJECT TO 0 <= ai <= C   for all i
    # sum(aiyi) = 0 for all i

    # KKT CONDITIONS
    # if ai = 0, f(xi) = yi(wT.x1 + b) >= 1
    # if ai = C, f(xi) = yi(wT.x1 + b) <= 1
    # if 0 < ai < C, f(xi) = yi(wT.x1 + b) = 1

    def __init__(self, DATA, DATA_HEADER, TARGET, TARGET_HEADER, K, ALPHAS, margin_type, C, alpha_seed,
                 max_passes, tol, SMO_a2_selection_method, conv_kill, pct_change, conv_end_method):
        # DATA MUST COME IN AS [ [] = COLUMNS ], HEADER MUST COME IN AS [[]]
        # TARGET MUST COME IN AS [[]], HEADER MUST COME IN AS [[]]
        # K (KERNEL MATRIX) MUST COME IN AS [ [] = ROWS ]



        if not isinstance(DATA, dict):
            self.DATA = n.array(DATA, dtype=n.float64).transpose()   # TURN DATA INTO [ [] = ROWS ]
        else:
            self.DATA = sd.core_sparse_transpose(DATA)
        self.DATA_HEADER = DATA_HEADER

        self.TARGET = n.array(TARGET[0], dtype=int)
        self.TARGET_HEADER = TARGET_HEADER

        self.K = K
        self.ALPHAS = ALPHAS
        self.KKT_BEST_ALPHAS_HOLDER = self.ALPHAS.copy()  # BELIEVING ALPHAS IS np SO CAN USE .copy() HERE
        self.margin_type = margin_type
        self.C = C

        self.tol = tol
        self.max_passes = max_passes
        self.SMO_a2_selection_method = SMO_a2_selection_method

        self.conv_kill = conv_kill
        self.pct_change = pct_change
        self.conv_end_method = conv_end_method

        # PLACEHOLDERS
        self.alpha_seed = alpha_seed
        self.alpha_handling()
        self.full_epoch_ctr = 0

        self.L = 0
        self.H = 0
        self.b = 0
        self.NON_BOUND_IDXS = []    # EXAMPLES FORMING THE MARGIN, I.E.,  0 < alpha < C
        self.BOUND_IDXS = []        # EXAMPLES IN THE BOUNDS, I.E., alpha = 0 (OUTSIDE THE RIVER) or alpha = C (INSIDE THE RIVER)
        self.F_CACHE = []
        self.ERROR_CACHE = []
        self.SUPPORT_VECTORS = n.array([], dtype=float)
        self.SUPPORT_TARGETS = n.array([], dtype=float)
        self.SUPPORT_ALPHAS = n.array([], dtype=float)
        self.SUPPORT_KERNELS = n.array([], dtype=float)
        self.passes = 0
        self.start_time = ''

    def return_fxn(self):
         return self.SUPPORT_VECTORS, self.SUPPORT_ALPHAS, self.SUPPORT_TARGETS, self.SUPPORT_KERNELS, self.b, \
                    self.ALPHAS, self.K, self.passes


    def use_early_stopping(self):
        '''Indicates whether early stopping for dev/_validation error.'''
        return False


    def early_stop_handling(self):
        '''Calculation of dev _validation errar for early stop.'''
        return None


    def alpha_handling(self):
        '''Allows (or disallows) user to choose to reset alphas or keep old values.'''
        if n.array_equiv(self.ALPHAS, []) or vui.validate_user_str(f'Reset ALPHAS? (y/n) > ', 'YN') == 'Y':
            self.ALPHAS = n.fromiter((self.alpha_seed for _ in range(len(self.TARGET))), dtype=float)
            # 10/16/22 BLEW UP DURING K-FOLD FOR NON_ZERO_ALPHAS OUT OF RANGE AGAINST DATA.  THINKING self.KKT_BEST_ALPHA_HOLDER
            # WAS NOT CORRECTLY SHAPED SO MODIFYING IT HERE ALONG WITH self.ALPHAS TO SEE IF IT FIXES IT
            self.KKT_BEST_ALPHAS_HOLDER = self.ALPHAS.copy()


    '''
    def objective_function(self):  # SEEMS LIKE THIS ISNT USED ANYWHERE (MAYBE TO TEST IMPROVEMENT WHEN SELECTING a2?)
        # W(a) = sum(all alphas) - 0.5 * sum over i, sum over j [ (yi * yj * ai * aj * k<xi,xj> ]
        W = n.sum(self.ALPHAS)
        # len(ALPHAS) = len(X) = len(Y) = len(K) = len(K[0])
        W += n.sum(n.outer(self.TARGET, self.TARGET) * n.outer(self.ALPHAS, self.ALPHAS) * self.K)
        return W
    '''


    def non_bound_idxs(self):
        # BEAR CHANGED THIS AS EXPERIMENT 9/23/22 FROM
        # return n.fromiter((_ for _ in range(len(self.ALPHAS)) if self.ALPHAS[_] > 0 and self.ALPHAS[_] < self.C), dtype=int)
        # TO
        return n.fromiter((_ for _ in range(len(self.ALPHAS)) if self.ALPHAS[_] > 0 and self.ALPHAS[_] <= self.C), dtype=int)


    def bound_idxs(self):
        # BEAR CHANGED THIS AS EXPERIMENT 9/23/22 FROM
        # return n.fromiter((_ for _ in range(len(self.ALPHAS)) if self.ALPHAS[_] == 0 or self.ALPHAS[_] == self.C), dtype=int)
        # TO
        return n.fromiter((_ for _ in range(len(self.ALPHAS)) if self.ALPHAS[_] == 0), dtype=int)


    def KKT_test(self, a_idx):  # RETURN TRUE IF SATISFIED, FALSE IF VIOLATED
        # F_CACHE IS BEING UPDATED ONLY WHEN SUCCESSFUL UPDATES TO a1 & a2
        functional_margin = self.TARGET[a_idx] * self.F_CACHE[a_idx]
        # if ai = 0, f(xi) = yi(wT.x1 + b) >= 1
        KKT1 = functional_margin > 1 + self.tol and self.ALPHAS[a_idx] == 0
        KKT2 = functional_margin < 1 - self.tol and self.ALPHAS[a_idx] == self.C
        KKT3 = (functional_margin >= 1 - self.tol and functional_margin <= 1 + self.tol) and \
               (self.ALPHAS[a_idx] > 0 and self.ALPHAS[a_idx] < self.C)

        return KKT1 or KKT2 or KKT3  # RETURNS True IF ANY OF THE 3 ARE TRUE, OTHERWISE False IF ALL ARE FALSE

    ''' THIS IS ONLY USEFUL FOR A LINEAR KERNEL, WHERE PREDICTED COULD BE CALCULATED FROM wx+b
    def w(self):
        w = n.matmul(self.ALPHAS.astype(float) * self.TARGET.astype(float), self.DATA.astype(float))
        return w
    '''
    ''' 
        #e Calculate individual example error for Platt error update rules.'''


    def F(self):
        # F Calculate function value for all examples.
        NZ = n.argwhere(self.ALPHAS != 0).transpose()[0]   # NZ = non-zero
        self.F_CACHE = n.matmul(self.ALPHAS[NZ] * self.TARGET[NZ], self.K[NZ].astype(float)) + self.b


    def f(self, ex_idx):
        # f Calculate function value for a single example.
        NZ = n.argwhere(self.ALPHAS != 0).transpose()[0]   # NZ = non-zero
        return n.matmul(self.ALPHAS[NZ] * self.TARGET[NZ], self.K[ex_idx][NZ].astype(float)) + self.b


    def E(self):
        # Calculate error for all examples.
        self.ERROR_CACHE = self.F_CACHE - self.TARGET


    def e(self, a_idx):
        # Calculate individual example error for Platt error update rules.
        return self.F_CACHE[a_idx] - self.TARGET[a_idx]


    def eta(self, a1_idx, a2_idx):
        # eta = SECOND DERIVATIVE OF THE OBJECTIVE FUNCTION, USUALLY (HOPEFULLY) LESS THAN ZERO
        return 2 * self.K[a1_idx][a2_idx] - self.K[a1_idx][a1_idx] - self.K[a2_idx][a2_idx]


    def L_H_bound_check(self, a1_idx, a2_idx):
        # CALCULATE THE BOUNDS L AND H FOR a2 (LIES BETWEEN 0 AND C)
        # WHEN y1 != y2:
        if self.TARGET[a2_idx] != self.TARGET[a1_idx]:
            self.L = max(0, self.ALPHAS[a2_idx] - self.ALPHAS[a1_idx])
            self.H = min(self.C, self.C + self.ALPHAS[a2_idx] - self.ALPHAS[a1_idx])
        # WHEN y1 == y2
        elif self.TARGET[a2_idx] == self.TARGET[a1_idx]:
            self.L = max(0, self.ALPHAS[a1_idx] + self.ALPHAS[a2_idx] - self.C)
            self.H = min(self.C, self.ALPHAS[a1_idx] + self.ALPHAS[a2_idx])
        # self.L & self.H ARE AUTOMATICALLY UPDATED, NO NEED TO RETURN, JUST RETURN GO OR NO-GO
        # IF THERE IS NO GAP, a2 CANNOT BE CHANGED, GO TO A NEW EXAMPLE
        if self.L < self.H:
            return True
        else:
            return False


    def joint_alpha_optimizer(self, a1_idx, a2_idx):
        # IF FAILS eta TEST, OR CHANGE TO a2 IS TOO SMALL, RETURNS False (MEANING FALSE THAT a2, a1, & b WERE UPDATED)
        # RESULTS ARE ONLY PUBLISHED TO a2, a1, & b IF PASSES THE eta TEST, OR BIG ENOUGH CHANGE TO a2 (RETURNS True)

        # CALCULATE eta BEFORE CALCULATING a2_new, THE SIGN OF eta (OR ZERO) DICTATES HOW a2_new IS TO BE CALCULATED
        eta_test = self.eta(a1_idx, a2_idx)
        if eta_test >= 0:
            # IF eta IS ZERO OR POSITIVE, CALCULATE a2 AT a2=L and a2=H, AND TAKE THE HIGHEST, IF BOTH ARE THE SAME,
            # NO PROGRESS CAN BE MADE, SO SKIP TO NEXT EXAMPLE
            # A FORMULA THAT ONLY DEPENDS ON a2 NEEDS TO BE USED HERE, BUT IT IS HUGE, AND HAS HUGE SUBCALCLUATIONS THAT
            # NEED TO BE DONE, SO COPPING OUT AND SKIPPING TO NEXT EXAMPLE
            # ONLY PROCEEDING IF eta (2ND DERIVATIVE OF OBJECTIVE) IS < 0.
            return False

        elif eta_test < 0:
            # IF eta IS NEGATIVE, 2ND DERIVATIVE OF OBJECTIVE IS NEGATIVE, INDICATING THERE IS A MAX FOR W BETWEEN L & H
            # CALCULATE a2_new, BUT KEEP a2_old, NEED IT TO CALCULATE a1_new, b, & UPDATE ERROR CACHE
            a2_old = deepcopy(self.ALPHAS[a2_idx])
            a2_new = self.ALPHAS[a2_idx] - self.TARGET[a2_idx] * (
                        self.ERROR_CACHE[a1_idx] - self.ERROR_CACHE[a2_idx]) / eta_test
            # CLIP a2_new IF LESS THAN L OR GREATER THAN H
            if a2_new >= self.H:
                a2_new = self.H
            elif a2_new <= self.L:
                a2_new = self.L
            # else: a2_new = a2_new     # IF BETWEEN L & H

            # IF CHANGE TO a2 IS TOO SMALL, FORGET IT AND GO TO NEXT a2, OTHERWISE UPDATE self.ALPHAS
            if n.abs(a2_new - self.ALPHAS[a2_idx]) < 1e-5:
                return False
            else:
                # CONSTRUCTION COMMENTARY
                # print(f'Acceptable a2 found at index {a2_idx}.  functional margin = {self.TARGET[a2_idx] * self.f(a2_idx)}, ALPHA = {self.ALPHAS[a2_idx]}, \n' \
                #               f'X = {self.DATA[a2_idx][0]}, Y = {self.DATA[a2_idx][1]}.  ')
                self.ALPHAS[a2_idx] = n.round(a2_new, 10)

        # CALCULATE a1_new, BUT KEEP a1_old, NEED IT TO CALCULATE b & UPDATE ERROR_CACHE.
        a1_old = deepcopy(self.ALPHAS[a1_idx])
        a1_new = a1_old + self.TARGET[a1_idx] * self.TARGET[a2_idx] * (a2_old - a2_new)
        self.ALPHAS[a1_idx] = n.round(a1_new, 10)

        # GET RID OF ANY ROUNDING ERROR THAT MAY HAPPEN TO ALPHAS (ALPHAS ALWAYS > 0)
        self.ALPHAS = n.abs(self.ALPHAS)

        # CALCULATE b
        # b_old = deepcopy(self.b)
        b1 = self.b - self.ERROR_CACHE[a1_idx] - self.TARGET[a1_idx] * (a1_new - a1_old) * self.K[a1_idx][a1_idx] - \
             self.TARGET[a2_idx] * (a2_new - a2_old) * self.K[a1_idx][a2_idx]
        b2 = self.b - self.ERROR_CACHE[a2_idx] - self.TARGET[a1_idx] * (a1_new - a1_old) * self.K[a1_idx][a2_idx] - \
             self.TARGET[a2_idx] * (a2_new - a2_old) * self.K[a2_idx][a2_idx]

        if a1_new > 0 and a1_new < self.C:
            self.b = b1
        elif a2_new > 0 and a2_new < self.C:
            self.b = b2
        else:
            self.b = (b1 + b2) / 2

        ''' BYPASSING THIS, CANT GET IT TO WORK.  SIMPLY UPDATING ERROR_CACH ON EVERY a1 VIOLATOR SELECTION, B4 OPTIMIZING
        # PLATT ERROR CACHE UPDATE RULE.  IF A NON-BOUND EXAMPLE IS INVOLVED IN OPTIMIZATION, SET ITS ERROR TO ZERO
        if a1_old > 0 and a1_old < self.C: self.ERROR_CACHE[a1_idx] = 0
        elif a2_old > 0 and a2_old < self.C: self.ERROR_CACHE[a2_idx] = 0
        # OTHERWISE, IF NOT IN OPTIMIZATION, USE FORMULA TO UPDATE
        for idx in self.non_bound_idxs():
            self.ERROR_CACHE[idx] = self.ERROR_CACHE[idx] + self.TARGET[a1_idx] * (a1_new - a1_old) * self.K[a1_idx][idx] + \
                                    self.TARGET[a2_idx] * (a2_new - a2_old) * self.K[a2_idx][idx] + b_old - self.b'''

        return True


    def build_boundary_objects(self):
        ''''Build self.SUPPORT_VECTORS, self.SUPPORT_ALPHAS, self.SUPPORT_TARGETS, self.SUPPORT_KERNELS'''

        # THIS IS CALCULATED IN 2 PLACES:
        # 1) AFTER ENTIRE SMO HAS RUN AND ENDED (MEANING FINALIZED ALPHAS ARE ESTABLISHED)
        # 2) FOR early_stop_handling IN SVMCoreSMOCode_MLPackage()... SUPPORT_VECTORS, SUPPORT_ALPHAS, SUPPORT_TARGETS ARE
        #    NEEDED TO CALCULATE THE NEW KERNEL NEEDED FOR CALCULATING DEV_DATA ERROR

        # AS OF 10/14/22 WAS USING self.ALPHAS, BUT CHANGING TO self.KKT_BEST_ALPHA_HOLDER.
        # IF RUNNING WITH DEV/VAL, WHEN TIME TO CHECK dev_error, LATEST PASS MAY BE COINCIDENTALLY ON A VERY BAD KKT VIOLATOR SPOT
        # (KKT COUNT MAY BE GYRATING) AND PREMATURELY END FOR DEV/VAL DIVERGENCE, SO USING BEST KKT ALPHAS TO PREVENT FALSE ABORTS
        # BUT THEN IF BEST_KKT_ALPHAS DOES NOT IMPROVE OVER NEXT DEV/VAL CHECKS, CAN ABORT FOR NO DEV IMPROVEMENT

        NON_ZERO_ALPHAS = n.argwhere(self.KKT_BEST_ALPHAS_HOLDER != 0).transpose()[0]

        # DATA IS [[]=ROWS]   .... BEAR, SHOULD BE ABLE TO USE MLRowColumnOperations HERE
        if isinstance(self.DATA, (list,tuple,n.ndarray)):
            self.SUPPORT_VECTORS = n.array(self.DATA, dtype=n.float64)[NON_ZERO_ALPHAS]
            self.SUPPORT_ALPHAS = self.KKT_BEST_ALPHAS_HOLDER[NON_ZERO_ALPHAS]
            self.SUPPORT_TARGETS = self.TARGET[NON_ZERO_ALPHAS]
        elif isinstance(self.DATA, dict):
            self.SUPPORT_VECTORS = n.empty((len(NON_ZERO_ALPHAS), sd.inner_len_quick(self.DATA)), dtype=object)
            self.SUPPORT_ALPHAS = n.empty((len(NON_ZERO_ALPHAS)), dtype=n.float64)
            self.SUPPORT_TARGETS = n.empty((len(NON_ZERO_ALPHAS)), dtype=n.int8)
            for sv_idx, data_idx in enumerate(NON_ZERO_ALPHAS):
                self.SUPPORT_VECTORS[sv_idx] = sd.unzip_to_ndarray({0:self.DATA[data_idx]})[0][0]
                self.SUPPORT_ALPHAS[sv_idx] = self.KKT_BEST_ALPHAS_HOLDER[data_idx]
                self.SUPPORT_TARGETS[sv_idx] = self.TARGET[data_idx]

        if isinstance(self.K, (list,tuple,n.ndarray)):
            self.SUPPORT_KERNELS = self.K[NON_ZERO_ALPHAS]
        elif isinstance(self.K, dict):
            self.SUPPORT_KERNELS = n.empty((len(NON_ZERO_ALPHAS), sd.outer_len(K)), dtype=object)
            for sv_idx, data_idx in enumerate(NON_ZERO_ALPHAS):
                self.SUPPORT_KERNELS[sv_idx] = sd.unzip_to_ndarray({0:self.K[data_idx]})[0][0]


    def run(self):

        st.show_start_time('SMO')
        self.start_time = datetime.now()  #.strftime("%m-%d-%Y %H:%M:%S")

        self.passes = 0

        # KKT CONVERGENCE HANDLING ###################################################################
        kkt_best_value = float('inf')  # USED TO FIND IF ALGORITHM IS GYRATING AROUND MINIMUM KKT VIOLATORS
        kkt_no_improv_ctr = 0

        full_epoch_mode = True  # The outer loop first iterates over the entire training set

        # INITIAL CALCULATION OF F_CACHE AND ERROR_CACHE
        self.F()
        self.E()

        while self.passes < self.max_passes:
            self.passes += 1
            print(
                f'\nRunning pass {self.passes} - ({"full epoch mode" if full_epoch_mode else "non-bound examples only"}) ...')
            KKT_violation_count = 0
            num_changed_alphas = 0  # WHEN NOT IN full_epoch_mode, the outer loop makes repeated passes over the non-bound
            # examples until all of the non-bound examples obey the KKT conditions within tol
            for a1_idx in range(len(self.ALPHAS)):  # OUTER alpha SELECTION LOOP
                '''
                The outer loop first iterates over the entire training set, determining whether each example violates the
                KKT conditions. After one pass through the training set, the outer loop iterates over only those examples 
                whose alphas are neither 0 nor C (the non-bound examples). Again, each example is checked against the KKT 
                conditions, and violating examples are eligible for immediate optimization and update. The outer loop 
                makes repeated passes over the non-bound examples until all of the non-bound examples obey the KKT 
                conditions within tol. The outer loop then iterates over the entire training set again. The outer loop 
                keeps alternating between single passes over the entire training set and multiple passes over the 
                non-bound subset until the entire training set obeys the KKT conditions within tol.  At that point, 
                the algorithm terminates.
                '''

                # MANAGE PLATT a1 SELECTION
                # 1) START WITH LOOKING IN FULL EPOCH FOR KKT VIOLATORS, & OPTIMIZE.
                # 2) ITERATE THRU ONLY NON-BOUND EXAMPLES LOOKING FOR KKT VIOLATORS & OPTIMIZE
                # 3) REPEAT STEP 2 UNTIL THERE NO KKT VIOLATORS IN NON-BOUND EXAMPLES
                # 4) GO BACK TO STEP 1
                # 5) REPEAT ALL OF THE ABOVE UNTIL THERE ARE NO VIOLATORS IN A FULL EPOCH PASS, THEN TERMINATE.

                # IF KKT CONDITIONS ARE SATISFIED, THEN CONTINUE UNTIL FINDING ANOTHER EXAMPLE THAT IS OUT OF CONSTRAINTS
                KKT_pass = self.KKT_test(a1_idx)
                if not full_epoch_mode and (
                        self.ALPHAS[a1_idx] == 0 or KKT_pass): continue   # or self.ALPHAS[a1_idx] == self.C
                if full_epoch_mode and KKT_pass:
                    continue
                elif full_epoch_mode and not KKT_pass:
                    KKT_violation_count += 1

                # CONSTRUCTION COMMENTARY
                # print(f'\nKKT violator at index {a1_idx} for a1.  functional margin = {self.TARGET[a1_idx] * self.f(a1_idx)}, ALPHA = {self.ALPHAS[a1_idx]}, \n' \
                #            f'X = {self.DATA[a1_idx][0]}, Y = {self.DATA[a1_idx][1]}.')

                if self.SMO_a2_selection_method == 'RANDOM':
                    a2_attempt_counter = 0
                    while True:
                        a2_idx = n.random.randint(0, len(self.ALPHAS))
                        if a2_idx == a1_idx: continue

                        if self.L_H_bound_check(a1_idx, a2_idx) is False:
                            a2_attempt_counter += 1
                        else:
                            if self.joint_alpha_optimizer(a1_idx, a2_idx) is False:
                                a2_attempt_counter += 1
                            else:
                                break

                        if a2_attempt_counter == 10: break

                    if a2_attempt_counter == 10: continue


                # PLATT a2 SELECTION ALGORITHM
                elif self.SMO_a2_selection_method == 'PLATT':
                    '''
                    Selection of 2nd alpha appears to be the same, whether first-alpha-selection is in full_epoch_mode or 
                    non_bound_mode. Ng has _random_ selection here in place of the Platt algorithm.  

                    Platt 2nd alpha algorithm:
                    (A) SMO chooses the second alpha to maximize the size of the step taken during joint optimization.
                    SMO approximates the step size by abs(E1 - E2) (the numerator in the a2 calculation). SMO keeps a cached 
                    error value E for every non-bound example in the training set and then chooses an error to approximately 
                    maximize the step size. If E1 is positive, SMO chooses an example with minimum error E2. If E1 is negative, 
                    SMO chooses an example with maximum error E2. 

                    Under unusual circumstances, SMO cannot make positive progress using the second choice heuristic described 
                    above. For example, positive progress cannot be made if the first and second training examples share identical 
                    input vectors, which causes the objective function to become flat along the direction of optimization.
                    To avoid this problem, SMO uses a hierarchy of second choice heuristics until it finds a pair of alphas
                    that can make positive progress. Positive progress can be determined by making a non-zero step upon joint 
                    optimization of the two alphas. 

                    The hierarchy of second choice heuristics consists of the following: 
                    (B) iterate through the non-bound examples, searching for a second example that can make positive progress
                    (C) iterate through the entire training set until an example is found that makes positive progress. 

                    Both (B) and (C) are started at _random_ locations in order not to bias SMO towards the examples at
                    the beginning of the training set. 

                    In extremely degenerate circumstances, none of the examples will make an adequate second example. 
                    When this happens, skip to another first example.

                    A cached error value E is kept for every example whose alpha is neither zero nor C. When alpha is 
                    non-bound and is involved in a joint optimization, its cached error is set to zero. Whenever a joint 
                    optimization occurs, the stored errors for all non-bound alphas that are not involved in the 
                    optimization are updated according to
                    E_k_new = E_k_old + y1(a1_new - a1_old) * K<x1,xk> + y2(a2_new - a2_old) * K(x2,xk) + b_old - b_new
                    (note --- this equation is for when using w.x - b)
                    When an error E is required by SMO, it will look up the error in the error cache if
                    the corresponding alpha is not at bound. Otherwise, it will evaluate
                    the current SVM decision function based on the current alpha vector.

                    '''

                    while True:
                        platt_rule_a_worked = False
                        platt_rule_b_worked = False
                        platt_rule_c_worked = False
                        # PLATT RULE A, SELECT BASED ON HIGHEST/LOWEST ERROR FOR NON-BOUND EXAMPLES IN ERROR CACHE
                        try:  # IS BLOWING UP ON FIRST PASS (DUE TO ALL w.x + b BEING ZERO I THINK) SO JUST PICK RANDOM IDX
                            # DONT LET a2_idx == a1_idx
                            if self.ERROR_CACHE[a1_idx] >= 0:
                                _ = n.fromiter((self.ERROR_CACHE[_] if (_ != a1_idx and _ in self.non_bound_idxs()) else float(
                                    'inf') for _ in range(len(self.ERROR_CACHE))), dtype=float)
                                a2_idx = n.argwhere(_ == n.min(_))[0][0]
                            elif self.ERROR_CACHE[a1_idx] < 0:
                                _ = n.fromiter((self.ERROR_CACHE[_] if (_ != a1_idx and _ in self.non_bound_idxs()) else float(
                                    '-inf') for _ in range(len(self.ERROR_CACHE))), dtype=float)
                                a2_idx = n.argwhere(_ == n.max(_))[0][0]
                            del _
                        except:
                            # CONSTRUCTION COMMENTARY
                            # print(f'EXCEPTION TRYING TO GET ERROR FOR PLATT RULE A DURING PASS {passes}.  CHOOSING a2_idx RANDOMLY FROM NON-BOUND')
                            while True:
                                a2_idx = n.random.choice(self.non_bound_idxs(), 1, False)[0]
                                if a2_idx != a1_idx: break

                        # IF FAILS L_H CHECK, SKIP TO RULE B, IF PASSES, RUN OPTIMIZER
                        if self.L_H_bound_check(a1_idx, a2_idx) is False:
                            pass
                            # CONSTRUCTION COMMENTARY
                            # print(f'a2={a2_idx} failed Platt Rule A on L-H gap tests.  Proceeding to Platt rule B.')
                        else:
                            # IF FAILS eta CHECK, OR CHANGE TO a2 IS TOO SMALL, RETURNS False, SKIP TO RULE B
                            if self.joint_alpha_optimizer(a1_idx, a2_idx) is False:
                                pass
                                # CONSTRUCTION COMMENTARY
                                # print(f'a2={a2_idx} failed Platt Rule A for eta >= 0 or change to a2 too small.  Proceeding to Platt rule B.')
                            # IF RETURNS True, optimizer RAN W/O SHORT-CIRCUIT AND a2, a1, & b WERE UPDATED.  BREAK & GO TO NEXT a1
                            else:
                                # CONSTRUCTION COMMENTARY
                                # print(f'a2, a1, and b successfully updated using Platt rule A for a1={a1_idx} and a2={a2_idx}')
                                platt_rule_a_worked = True
                                break

                        # PLATT RULE B, RANDOMLY SELECT FROM ALL NON-BOUND EXAMPLES (SELECT FROM EXAMPLES "IN THE RIVER") UNTIL
                        # PROGRESS CAN BE MADE
                        _ = self.non_bound_idxs()
                        try:  # EXCEPTION WHEN EMPTY non_bound_idxs(), EMPTY LIST BLOWS UP choice, IF EXCEPT, SKIP TO RULE (C)
                            __ = n.random.choice([i for i in _ if i != a1_idx], len(_) - 1, False)
                            for a2_idx in __:
                                # IF FAILS H-L CHECK, SKIP TO NEXT RANDOM NON-BOUND EXAMPLE, IF PASSES, RUN OPTIMIZER
                                if self.L_H_bound_check(a1_idx, a2_idx) is False:
                                    # CONSTRUCTION COMMENTARY
                                    # print(f'a2={a2_idx} failed Platt Rule B on L-H gap tests.  Proceeding to next _random_ non-bound a2 idx.')
                                    continue
                                else:
                                    # IF FAILS eta CHECK, OR CHANGE TO a2 IS TOO SMALL, RETURNS False, SKIP TO NEXT RANDOM NON-BOUND EXAMPLE
                                    if self.joint_alpha_optimizer(a1_idx, a2_idx) is False:
                                        # CONSTRUCTION COMMENTARY
                                        # print(f'a2={a2_idx} failed Platt Rule B for eta >= 0 or change to a2 too small.  Proceeding to next _random_ non-bound a2 idx.')
                                        continue
                                    # IF RETURNS True, optimizer RAN W/O SHORT-CIRCUIT AND a2, a1, & b WERE UPDATED.  BREAK & GO TO NEXT a1
                                    else:
                                        platt_rule_b_worked = True
                                        # CONSTRUCTION COMMENTARY
                                        # print(f'a2, a1, and b successfully updated using Platt rule B for a1={a1_idx} and a2={a2_idx}')
                                        break
                            # else: CONSTRUCTION COMMENTARY print(f'Failed to update a2, a1, and b using Platt rule B.  Proceeding to Platt rule C.')
                        except:
                            pass
                            # CONSTRUCTION COMMENTARY print(f'Platt rule B failed for no non-bound examples.  Proceeding to Platt rule C.')

                        if platt_rule_b_worked: break  # IF UPDATED a2, a1, & b W/ PLATT RULE B, BREAK TO NEXT a1, ELSE GO TO PLATT RULE C

                        # PLATT RULE C, RANDOMLY SELECT FROM ALL BOUND EXAMPLES (EXAMPLES NOT "IN THE RIVER") UNTIL PROGRESS IS MADE
                        # DONT NEED TO RECORD "platt_rule_c_worked, True/False" HERE, BREAK a2 SELECTION LOOP NO MATTER IF WORKED OR NOT
                        _ = self.bound_idxs()
                        __ = n.random.choice([i for i in _ if i != a1_idx], len(_) - 1, False)
                        for a2_idx in __:
                            # IF FAILS H-L CHECK, SKIP TO NEXT RANDOM BOUND EXAMPLE, IF PASSES, RUN OPTIMIZER
                            if self.L_H_bound_check(a1_idx, a2_idx) is False:
                                pass
                                # CONSTRUCTION COMMENTARY print(f'a2={a2_idx} failed Platt Rule C on L-H gap tests.  Proceeding to next _random_ bound a2 idx.')
                                continue
                            else:
                                # IF FAILS eta CHECK, OR CHANGE TO a2 IS TOO SMALL, RETURNS False, SKIP TO NEXT RANDOM NON-BOUND EXAMPLE
                                if self.joint_alpha_optimizer(a1_idx, a2_idx) is False:
                                    # CONSTRUCTION COMMENTARY print(f'a2={a2_idx} failed Platt Rule C for eta >= 0 or change to a2 too small.  Proceeding to next _random_ bound a2 idx.')
                                    continue
                                # IF RETURNS True, optimizer RAN W/O SHORT-CIRCUIT AND a2, a1, & b WERE UPDATED.  BREAK & GO TO NEXT a1
                                else:
                                    # CONSTRUCTION COMMENTARY print(f'a2, a1, and b successfully updated using Platt rule C for a1={a1_idx} and a2={a2_idx}')
                                    platt_rule_c_worked = True
                                    break
                        else:
                            # CONSTRUCTION COMMENTARY print(f'Failed to update a2, a1, and b using Platt rule C.  Proceeding to next a1.')
                            break

                        break

                    if platt_rule_a_worked or platt_rule_b_worked or platt_rule_c_worked: pass
                    else: continue

                else:
                    raise ValueError(f'INVALID SMO a2 SELECTION ALGORITHM IN SVMCoreSMOCode.run().  MUST BE "RANDOM" OR "PLATT".')

                # IF REACHED THIS POINT, ALPHAS WERE SUCCESSFULLY UPDATED, SO INCREMENT num_changed_alphas IF full_epoch_mode IS False,
                # UPDATE ALL FUNCTION VALUES AND ERRORS
                self.F()
                self.E()

                if full_epoch_mode is False: num_changed_alphas += 1


            # MANAGE KKT CONVERGENCE & DEV/VAL CONVERGENCE ###########################################################################
            if full_epoch_mode:  # ONLY IMPLEMENT FOR full_epoch_mode, WHEN THERE IS A FULL COUNT FOR KKT_violation_count

                self.full_epoch_ctr += 1

                kkt_no_improv_ctr, self.conv_kill, kkt_best_value, kkt_abort = ni.NoImprov(KKT_violation_count,
                    self.full_epoch_ctr, 1, kkt_no_improv_ctr, self.conv_kill, kkt_best_value, self.pct_change,
                    'SMO KKT CONVERGENCE', conv_end_method=self.conv_end_method).min()

                if kkt_no_improv_ctr == 0:   # IF kkt_no_improv_ctr IS 0, MEANS JUST HIT A NEW BEST CASE

                    self.KKT_BEST_ALPHAS_HOLDER = self.ALPHAS.copy()

                    if self.use_early_stopping() is True:
                        # EARLY STOP HANDLING #####################################################################################
                        if self.early_stop_handling() == 'BREAK':
                            print(f'\n*** Training cycle stopped early on pass {self.passes} of {self.max_passes} for '
                                  f'divergence of _validation error. ***\n')

                            break
                        # END EARLY STOP HANDLING #####################################################################################

                if kkt_abort:
                    print(f'\nNumber of KKT violators has not improved beyond {kkt_best_value} for {kkt_no_improv_ctr} full epoch '
                          f'passes after {self.passes} of {self.max_passes} passes. Automatically ending SMO algorithm.')

                    print('\nStart time: ', self.start_time.strftime("%m-%d-%Y %H:%M:%S"))
                    st.show_end_time('SVM')
                    end_time = datetime.now()  # .strftime("%m-%d-%Y %H:%M:%S")
                    try: print(f'Elapsed time: {round((end_time - self.start_time).seconds / 60, 2)} minutes')
                    except: print('Subtracting end and start times is giving error')

                    break

            # END MANAGE KKT CONVERGENCE KILL #####################################################################################

            # DONT CHANGE THE ORDER OF THESE if/elifs, passes MUST BE FIRST
            if self.passes == self.max_passes:  # SKIP ELIFS BELOW IF ON LAST PASS TO PRESERVE full_epoch or not full_epoch STATE
                break

            # IF MADE A PASS THRU NOT IN full_epoch_mode AND NO ALPHAS WERE UPDATED, THEN all of the non-bound
            # examples obey the KKT conditions within tol SO GO BACK TO full_epoch_mode
            elif full_epoch_mode is False and num_changed_alphas == 0:
                full_epoch_mode = True

            # IF COMPLETED ONE PASS IN full_epoch_mode WITH VIOLATORS, CHANGE OVER TO non_bound_mode
            elif full_epoch_mode and KKT_violation_count > 0:
                print(f'\n*** {KKT_violation_count} KKT VIOLATORS AFTER LAST PASS IN full_epoch_mode ***')
                full_epoch_mode = False

            elif full_epoch_mode and KKT_violation_count == 0:
                print(f'\n*** SMO ALGORITHM TERMINATED AFTER {self.passes} PASSES FOR NO KKT VIOLATORS AFTER ONE FULL '
                      f'PASS IN full_epoch_mode ***\n')
                break



        # BECAUSE KKT_BEST IS NOW USED FOR FINAL SUPPORT OBJECTS CALC
        if self.use_early_stopping() is True:
            self.KKT_BEST_ALPHAS_HOLDER = self.DEV_BEST_ALPHAS_HOLDER.copy()
            # OTHERWISE WITH0UT early_stopping FOR DEV/VAL, JUST CARRY FORWARD self.KKT_BEST_ALPHAS_HOLDER

        # BECAUSE self.ALPHAS IS BEING RETURNED FROM THE MODULE
        self.ALPHAS = self.KKT_BEST_ALPHAS_HOLDER.copy()

        # BUILD BOUNDARDY INFO OBJECTS
        self.build_boundary_objects()

        # USING a * y * K, IF USING LINEAR KERNEL COULD USE wx+b WHERE w = SUM(a*y*x)
        PREDICTED = n.matmul(self.TARGET * self.ALPHAS, self.K.astype(float)) + self.b

        # PREVIEW OF RESULTS ######################################################################################################
        '''
        print(f'\nPREVIEW OF RESULTS')
        rows = 20
        OUTPUT = p.DataFrame(
                            {self.TARGET_HEADER[0][0]: self.TARGET[:rows], 'PREDICTED': PREDICTED[:rows],
                             'CLASS': lf.link_fxns(PREDICTED, 'SVM_Perceptron')[:rows], 'ALPHAS': self.ALPHAS[:rows]} |
                            {self.DATA_HEADER[0][_]:__[:rows] for _,__ in enumerate(self.DATA.transpose()[:10])}
        )
        print(OUTPUT)
        '''
        # END PREVIEW OF RESULTS ######################################################################################################

        # COUNT ALL KKT VIOLATORS AFTER LAST PASS FOR REPORTING (LAST PASS MAY HAVE BEEN NON-BOUND ONLY)
        self.F()
        self.E()
        KKT_violation_count = 0
        for ex_idx in range(len(self.ALPHAS)):
            if not self.KKT_test(ex_idx):  # IF NOT PASS KKT TEST
                KKT_violation_count += 1

        if self.passes == self.max_passes:
            print(f'\n*** SMO COMPLETED {self.passes} OF {self.max_passes} ALLOWED PASSES, WITH {KKT_violation_count} KKT VIOLATOR(S) IN {len(self.K)} EXAMPLES. ***')
        elif self.passes < self.max_passes:
            print(f'\n*** SMO COMPLETE, PERFORMING {self.passes} OF {self.max_passes} ALLOWED PASSES, WITH {KKT_violation_count} KKT VIOLATORS. ***')

        # TEST PRINTING DATA & BOUNDARY, ONLY FOR IF DATA IS 2D
        # svmg2d.svm_data_and_results_2d(self.TARGET, self.DATA.transpose(), self.SUPPORT_VECTORS, self.SUPPORT_ALPHAS,
        #                                self.SUPPORT_TARGETS, self.b)

        return self.return_fxn()



class SVMCoreSMOCode_MLPackage(SVMCoreSMOCode):

    def __init__(self, DATA, DATA_HEADER, TARGET, TARGET_HEADER, DEV_DATA, DEV_TARGET, K, ALPHAS, margin_type, C,
                 alpha_seed, cost_fxn, kernel_fxn, constant, exponent, sigma, max_passes, tol,
                 SMO_a2_selection_method, early_stop_interval, conv_kill, pct_change, conv_end_method):

        # DATA MUST COME IN AS [ [] = COLUMNS ], HEADER MUST COME IN AS [[]]
        # TARGET MUST COME IN AS [[]], HEADER MUST COME IN AS [[]]
        # K (KERNEL MATRIX) MUST COME IN AS [ [] = ROWS ]

        super().__init__(DATA, DATA_HEADER, TARGET, TARGET_HEADER, K, ALPHAS, margin_type, C, alpha_seed,
                 max_passes, tol, SMO_a2_selection_method, conv_kill, pct_change, conv_end_method)

        self.DEV_DATA = DEV_DATA
        self.DEV_TARGET = DEV_TARGET
        self.cost_fxn = cost_fxn
        self.kernel_fxn = kernel_fxn
        self.constant = constant
        self.exponent = exponent
        self.sigma = sigma

        # DEV/VAL CONVERGENCE HANDLING ###################################################################
        # self.early_stop_error_holder ---- DEFUNCT
        self.early_stop_interval = early_stop_interval
        self.DEV_BEST_ALPHAS_HOLDER = self.ALPHAS.copy()
        self.dev_best_value = float('inf')
        self.dev_no_improv_ctr = 0


    def use_early_stopping(self):
        '''Indicates whether early stopping for dev/_validation error.'''
        if self.early_stop_interval == 1e12:    # 1e12 IS THE DUMMY NUMBER USED IN MLRunTemplate TO TURN OFF early_stopping()
            return False
        else: return True


    def early_stop_handling(self):
        '''Calculation of dev _validation errar for early stop.'''

        self.build_boundary_objects()  # CALCULATING THIS TO GET SUPPORT_VECTORS, SUPPORT_TARGETS, SUPPORT_ALPHAS

        DEV_OUTPUT_VECTOR = svmoc.svm_dev_test_output_calc(self.SUPPORT_VECTORS, self.SUPPORT_TARGETS,
                   self.SUPPORT_ALPHAS, self.b, self.DEV_DATA, self.kernel_fxn, self.constant, self.exponent, self.sigma)

        dev_error = sec.svm_error_calc(self.DEV_TARGET, DEV_OUTPUT_VECTOR, self.cost_fxn, 0)

        print(f'\nPASS {self.passes} VALIDATION ERROR = {dev_error}')

        # GIVE dev_error 3 CHANCES TO IMPROVE WHILE KKT_violators IS STILL IMPROVING
        self.dev_no_improv_ctr, DUM, self.dev_best_value, dev_abort = ni.NoImprov(dev_error, self.full_epoch_ctr, 1,
                      self.dev_no_improv_ctr, 5, self.dev_best_value, 0,
                      'SMO DEV/VAL early_stop_convergence', conv_end_method="KILL").min()

        if self.dev_no_improv_ctr == 0:  # IF self.dev_no_improv_ctr IS 0, MEANS JUST HIT A NEW BEST CASE
            self.DEV_BEST_ALPHAS_HOLDER = self.ALPHAS.copy()  # BELIEVING ALPHAS IS np, SO CAN USE .copy()

        if dev_abort:
            print(f'\nDev/val error diverged after {self.passes} of {self.max_passes} passes. '
                  f'Automatically ending SMO algorithm.')

            print('\nStart time: ', self.start_time.strftime("%m-%d-%Y %H:%M:%S"))
            st.show_end_time('SVM')
            end_time = datetime.now()  # .strftime("%m-%d-%Y %H:%M:%S")
            try:
                print(f'Elapsed time: {round((end_time - self.start_time).seconds / 60, 2)} minutes')
            except:
                print('Subtracting end and start times is giving error')

            return 'BREAK'


    def alpha_handling(self):
        '''Allows (or disallows) user to choose to reset alphas or keep old values.'''
        # 7-19-22 REMOVING CONDITIONAL SO THAT ALPHAS ARE GENERATED EVERY TIME THIS FXN IS CALLED, THIS ENSURES ALPHAS HAS THE SAME
        # LEN AS THE PARTITION (PREVIOUSLY ALPHAS WAS ONLY BEING GENERATED ON THE FIRST PASS, WHEN ALPHAS == []), THEN
        # SIMPLY PASSING THRU OVER AND OVER WITHOUT REVISIT, CAUSING BROADCAST ERRORS
        self.ALPHAS = n.fromiter((self.alpha_seed for _ in range(len(self.TARGET))), dtype=float)
        # 10/16/22 PARENT BLEW UP DURING K-FOLD FOR NON_ZERO_ALPHAS OUT OF RANGE AGAINST DATA.  THINKING
        # self.KKT_BEST_ALPHA_HOLDER WAS NOT CORRECTLY SHAPED SO MODIFIED IT WITH self.ALPHAS TO SEE IF IT FIXES IT,
        # APPLYING SAME TYPE OF BAND-AID HERE FOR self.DEV_BEST_ALPHAS_HOLDER JUST IN CASE, NOT SURE IF NEEDED
        self.KKT_BEST_ALPHAS_HOLDER = self.ALPHAS.copy()
        self.DEV_BEST_ALPHAS_HOLDER = self.ALPHAS.copy()







if __name__ == '__main__':

    from ML_PACKAGE.SVM_PACKAGE import KernelBuilder as kb
    from ML_PACKAGE.SVM_PACKAGE.print_results import svm_graph_2d as svmg2d

    print(f'\nBuilding DATA and TARGET...')

    DATA = [[], []]
    DATA_HEADER = [['X', 'Y']]
    TARGET = [[]]
    TARGET_HEADER = [['TARGET']]
    gap = 5
    gap_slope = -1
    gap_y_int = -5
    length = 200
    while True:
        x = 10 * n.random.randn()
        y = 10 * n.random.randn()
        if y > gap_slope * x + (gap_y_int - 0.5 * gap) and y < gap_slope * x + (gap_y_int + 0.5 * gap): continue
        DATA[0].append(x)
        DATA[1].append(y)
        if y < gap_slope * x + (gap_y_int - 0.5 * gap): TARGET[0].append(-1)
        if y > gap_slope * x + (gap_y_int + 0.5 * gap): TARGET[0].append(1)
        if len(DATA[0]) == length: break

    # TEST OF "Y=constant" BOUNDARY
    # DATA = [[1,2,3], [1,1.5,1]]
    # DATA_HEADER = [['X', 'Y']]
    # TARGET = [[-1, 1, -1]]
    # TARGET_HEADER = [['TARGET']]

    DATA = n.array(DATA, dtype=float)
    print(f'Done.')

    print(f'\nTransforming DATA to sparse_dict...')
    DATA = sd.zip_list(DATA)
    print(f'Done.')

    print(f'\nGoing into svm_data_2d...')
    svmg2d.svm_data_2d(TARGET, DATA)

    print(f'\nTransposing sparse DATA')
    DATA = sd.sparse_transpose(DATA)
    print(f'Done.')

    # DATA MUST BE [] = ROWS FOR KernelBuilder
    K = kb.KernelBuilder(DATA, kernel_fxn='LINEAR', prompt_bypass=True).build()
    print(f'KERNEL DONE.')

    print(f'\nTransposing sparse DATA')
    DATA = sd.sparse_transpose(DATA)
    print(f'Done.')

    ALPHAS = []
    margin_type = 'SOFT'
    C = float('inf')
    alpha_seed = 0
    max_passes = 10000
    tol = .001
    conv_kill = 500
    pct_change = 0
    conv_end_method = 'KILL'

    ####################################################################################################################################
    ####################################################################################################################################
    # RANDOM ###########################################################################################################################
    SMO_a2_selection_method = 'RANDOM'
    import time

    t1 = time.time()
    SUPPORT_VECTORS, SUPPORT_ALPHAS, SUPPORT_TARGETS, SUPPORT_KERNELS, b, ALPHAS, K, passes = \
    SVMCoreSMOCode(DATA, DATA_HEADER, TARGET, TARGET_HEADER, K, ALPHAS, margin_type, C, alpha_seed, max_passes,
                                  tol, SMO_a2_selection_method, conv_kill, pct_change, conv_end_method).run()

    t2 = time.time()
    print(f'\nb = ', b)

    svmg2d.svm_data_and_results_2d(TARGET, DATA, SUPPORT_VECTORS, SUPPORT_ALPHAS, SUPPORT_TARGETS, b)

    print(f'Random Done.')
    # END RANDOM #######################################################################################################################
    ####################################################################################################################################
    ####################################################################################################################################


    ####################################################################################################################################
    ####################################################################################################################################
    # PLATT ############################################################################################################################
    SMO_a2_selection_method = 'PLATT'

    t1 = time.time()
    SUPPORT_VECTORS, SUPPORT_ALPHAS, SUPPORT_TARGETS, SUPPORT_KERNELS, b, ALPHAS, K, passes = \
                SVMCoreSMOCode(DATA, DATA_HEADER, TARGET, TARGET_HEADER, K, ALPHAS, margin_type, C, alpha_seed, max_passes,
                tol, SMO_a2_selection_method, conv_kill, pct_change, conv_end_method).run()
    t2 = time.time()
    print(f'\nb = ', b)

    svmg2d.svm_data_and_results_2d(TARGET, DATA, SUPPORT_VECTORS, SUPPORT_ALPHAS, SUPPORT_TARGETS, b)

    print(f'PLATT Done.')
    # END PLATT ########################################################################################################################
    ####################################################################################################################################
    ####################################################################################################################################











