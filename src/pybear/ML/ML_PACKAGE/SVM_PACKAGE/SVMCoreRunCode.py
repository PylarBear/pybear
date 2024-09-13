import numpy as np
from data_validation import validate_user_input as vui
import sparse_dict as sd
from ML_PACKAGE.SVM_PACKAGE import KernelBuilder as kb
from ML_PACKAGE.SVM_PACKAGE.alpha_selection_algs import SVMCoreSMOCode as scsc, SVMCoreChunkCode as sccc, SVMCoreOsunaCode as scoc
from MLObjects.SupportObjects import master_support_object_dict as msod


class SVMCoreRunCode:

    # DATA MUST COME IN AS [ [] = COLUMNS ]
    # TARGET MUST COME IN AS [ [] = COLUMNS ]


    def __init__(self, DATA, TARGET, WORKING_SUPOBJS, K, ALPHAS, kernel_fxn, constant, exponent, sigma, margin_type,
            C, alpha_seed, alpha_selection_alg, max_passes, tol, SMO_a2_selection_method, conv_kill, pct_change, conv_end_method):

        self.DATA = DATA
        self.DATA_HEADER = WORKING_SUPOBJS[0][msod.QUICK_POSN_DICT()["HEADER"]].reshape((1,-1))

        # CHECK TARGET IS SINGLE LABEL, 1 & -1 ONLY
        if len(TARGET) > 1:
            raise ValueError(f"\n *** TARGET VECTOR GOING INTO SVMCoreRunCode is MULTI-CLASS. ***\n")
        if True in map(lambda x: x not in [1, -1], TARGET[0]):
            raise ValueError(f"\n *** TARGET VECTOR GOING INTO SVMCoreRunCode IS NOT {-1, 1} ONLY. ***\n")
        self.TARGET = TARGET
        self.TARGET_HEADER = WORKING_SUPOBJS[1][msod.QUICK_POSN_DICT()["HEADER"]].reshape((1,-1))

        # TO ALLOW FOR RE-INGESTION & CONTINUATION OF A TRAINING SESSION
        self.K = K
        self.ALPHAS = ALPHAS

        self.kernel_fxn = kernel_fxn.upper()
        self.constant = constant
        self.exponent = exponent
        self.sigma = sigma

        self.margin_type = margin_type.upper()
        self.C = C

        self.alpha_seed = alpha_seed
        self.alpha_selection_alg = alpha_selection_alg.upper()
        self.max_passes = max_passes
        self.tol = tol
        self.SMO_a2_selection_method = SMO_a2_selection_method.upper()
        self.conv_kill = conv_kill
        self.pct_change = pct_change
        self.conv_end_method = conv_end_method

        # PLACEHOLDERS
        self.PREDICTIONS = []
        self.SUPPORT_VECTORS = []
        self.SUPPORT_ALPHAS = []
        self.SUPPORT_TARGETS = []
        self.SUPPORT_KERNELS = []
        self.b = 0
        self.passes = 0


    def kernel_handling(self):
        if len(self.K) != len(self.DATA[0]) or \
            vui.validate_user_str(f'Recalculate kernel? (y/n) (need to do this if kernel function was changed '
                                     f'within this session) > ', 'YN') == 'Y':
            if isinstance(self.DATA, (list, tuple, np.ndarray)):
                self.K, self.run_data_as, self.return_kernel_as = kb.KernelBuilder(self.DATA.transpose(),
                    kernel_fxn=self.kernel_fxn, constant=self.constant, exponent=self.exponent, sigma=self.sigma,
                    DATA_HEADER=self.DATA_HEADER, prompt_bypass=False).build()
            elif isinstance(self.DATA, dict):
                self.K, self.run_data_as, self.return_kernel_as = kb.KernelBuilder(sd.sparse_transpose(self.DATA),
                    kernel_fxn=self.kernel_fxn, constant=self.constant, exponent=self.exponent, sigma=self.sigma,
                    DATA_HEADER=self.DATA_HEADER, prompt_bypass=False).build()


    def return_fxn(self):
        return self.SUPPORT_VECTORS, self.SUPPORT_ALPHAS, self.SUPPORT_TARGETS, self.SUPPORT_KERNELS, self.b, \
               self.ALPHAS, self.K, self.passes   # MUST RETURN K HERE BECAUSE K IS CREATED IN THIS MODULE


    def SMO_core_run_code(self):
        self.SUPPORT_VECTORS, self.SUPPORT_ALPHAS, self.SUPPORT_TARGETS, self.SUPPORT_KERNELS, self.b, self.ALPHAS, \
        self.K, self.passes = scsc.SVMCoreSMOCode(self.DATA, self.DATA_HEADER, self.TARGET, self.TARGET_HEADER,
        self.K, self.ALPHAS, self.margin_type, self.C, self.alpha_seed, self.max_passes, self.tol,
        self.SMO_a2_selection_method, self.conv_kill, self.pct_change, self.conv_end_method).run()


    def run(self):

        if self.K == [[]]:   # IF K IS EMPTY, BUILD, OTHERWISE kernel_handling()

            if isinstance(self.DATA, (list, tuple, np.ndarray)):
                self.K, self.run_data_as, self.return_kernel_as = kb.KernelBuilder(self.DATA.transpose(),
                    kernel_fxn=self.kernel_fxn, constant=self.constant, exponent=self.exponent, sigma=self.sigma,
                    DATA_HEADER=self.DATA_HEADER, prompt_bypass=False).build()
            elif isinstance(self.DATA, dict):
                self.K, self.run_data_as, self.return_kernel_as = kb.KernelBuilder(sd.sparse_transpose(self.DATA),
                    kernel_fxn=self.kernel_fxn, constant=self.constant, exponent=self.exponent, sigma=self.sigma,
                    DATA_HEADER=self.DATA_HEADER, prompt_bypass=False).build()
        else:
            self.kernel_handling()

        if self.alpha_selection_alg == 'SMO':
            self.SMO_core_run_code()

        elif self.alpha_selection_alg == 'CHUNK':  # CURRENTLY IMPOSSIBLE TO GET HERE, OPTION IS IMPOSSIBLE IN CoreConfig
            self.SUPPORT_VECTORS, self.SUPPORT_ALPHAS, self.SUPPORT_TARGETS, self.SUPPORT_KERNELS, self.b, self.ALPHAS, \
                self.K, self.passes = sccc.SVMCoreChunkCode()

        elif self.alpha_selection_alg == 'OSUNA':  # CURRENTLY IMPOSSIBLE TO GET HERE, OPTION IS IMPOSSIBLE IN CoreConfig
            self.SUPPORT_VECTORS, self.SUPPORT_ALPHAS, self.SUPPORT_TARGETS, self.SUPPORT_KERNELS, self.b, self.ALPHAS, \
                self.K, self.passes = scoc.SVMCoreOsunaCode()

        return self.return_fxn()





class SVMCoreRunCode_MLPackage(SVMCoreRunCode):
    # DATA MUST COME IN AS [ [] = COLUMNS ]
    # TARGET MUST COME IN AS [ [] = COLUMNS ]

    def __init__(self, TRAIN_SWNL, DEV_SWNL, WORKING_SUPOBJS, K, ALPHAS, kernel_fxn, constant,
                 exponent, sigma, margin_type, C, alpha_seed, cost_fxn, alpha_selection_alg, max_passes, tol,
                 SMO_a2_selection_method, early_stop_interval, conv_kill, pct_change, conv_end_method, run_data_as,
                 return_kernel_as, rebuild_kernel):

        super().__init__(TRAIN_SWNL[0], TRAIN_SWNL[1], WORKING_SUPOBJS, K, ALPHAS, kernel_fxn, constant, exponent, sigma,
                         margin_type, C, alpha_seed, alpha_selection_alg, max_passes, tol, SMO_a2_selection_method,
                         conv_kill, pct_change, conv_end_method)

        self.DEV_DATA = DEV_SWNL[0]
        self.DEV_TARGET = DEV_SWNL[1]
        self.cost_fxn = cost_fxn
        self.early_stop_interval = early_stop_interval
        self.early_stop_error_holder = float('inf')
        self.run_data_as = run_data_as
        self.return_kernel_as = return_kernel_as
        self.rebuild_kernel = rebuild_kernel


    def kernel_handling(self):
        if self.rebuild_kernel is True:
            if isinstance(self.DATA, (list, tuple, np.ndarray)):
                self.K = kb.KernelBuilder(self.DATA.transpose(), kernel_fxn=self.kernel_fxn, constant=self.constant,
                      exponent=self.exponent, sigma=self.sigma, DATA_HEADER=self.DATA_HEADER, prompt_bypass=True,
                      run_data_as=self.run_data_as, return_kernel_as=self.return_kernel_as).build()
            elif isinstance(self.DATA, dict):
                self.K = kb.KernelBuilder(sd.sparse_transpose(self.DATA), kernel_fxn=self.kernel_fxn, constant=self.constant,
                      exponent=self.exponent, sigma=self.sigma, DATA_HEADER=self.DATA_HEADER, prompt_bypass=True,
                      run_data_as=self.run_data_as, return_kernel_as=self.return_kernel_as).build()
        else:
            pass    # JUST LEAVE K AS WAS INGESTED

    def SMO_core_run_code(self):
        self.SUPPORT_VECTORS, self.SUPPORT_ALPHAS, self.SUPPORT_TARGETS, self.SUPPORT_KERNELS, self.b, self.ALPHAS, \
        self.K, self.passes = scsc.SVMCoreSMOCode_MLPackage(self.DATA, self.DATA_HEADER, self.TARGET, self.TARGET_HEADER,
        self.DEV_DATA, self.DEV_TARGET, self.K, self.ALPHAS, self.margin_type, self.C, self.alpha_seed, self.cost_fxn,
        self.kernel_fxn, self.constant, self.exponent, self.sigma, self.max_passes, self.tol,
        self.SMO_a2_selection_method, self.early_stop_interval, self.conv_kill, self.pct_change, self.conv_end_method).run()


    def return_fxn(self):
        # MUST RETURN K HERE BECAUSE K IS CREATED IN THIS MODULE
        return self.SUPPORT_VECTORS, self.SUPPORT_ALPHAS, self.SUPPORT_TARGETS, self.SUPPORT_KERNELS, self.b, \
               self.ALPHAS, self.K, self.passes, self.run_data_as, self.return_kernel_as



if __name__ == '__main__':
    pass

