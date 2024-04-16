import numpy as n, pandas as p
from ML_PACKAGE.GENERIC_PRINT import SummaryStatisticsTemplate as sst
from ML_PACKAGE.SVM_PACKAGE import link_fxns as lf


# INHERITED BY ALL ##################################################################################################
# conditional_print_for_all()   # Define what to report based on what the output will be as determined by the final link function
# always_printed_for_all()      # Statistics printed for all ML packages
# OVERWRITTEN IN CHILD ##############################################################################################
# rsq()                         # Define function for rsq
# iteration_print()             # Available for use by ML packages that have iterations
# ml_specific_print()           # Print instructions for specific ML module
# print()                       # Build printout for specific ML module by piecing together


class SVMSummaryStatisticsPrint(sst.SummaryStatisticsPrintTemplate):

    def rsq(self):
        # Define function for rsq
        return n.corrcoef(lf.link_fxns(self.OUTPUT_VECTOR[0], 'SVM_Perceptron').tolist(),
                          self.TARGET_VECTOR[0].tolist())[0][1]**2


    def __init__(self, OUTPUT_VECTOR, TARGET, TARGET_HEADER, ALPHAS, SUPPORT_VECTORS, SUPPORT_TARGETS, SUPPORT_ALPHAS, b,
                 error, passes, max_passes):

        self.TARGET_HEADER = TARGET_HEADER
        self.ALPHAS = ALPHAS
        self.SUPPORT_VECTORS = SUPPORT_VECTORS
        self.SUPPORT_TARGETS = SUPPORT_TARGETS
        self.SUPPORT_ALPHAS = SUPPORT_ALPHAS
        self.b = b
        self.passes = passes
        self.max_passes = max_passes

        self.link = 'SVM_Perceptron'

        super().__init__(OUTPUT_VECTOR, TARGET, self.link, error)


    def iteration_print(self):
        # Available for use by ML packages that have iterations
        print(f'\nAfter pass {self.passes} of {self.max_passes}:')
        pass


    def ml_specific_print(self):
        # Print instructions for specific ML module

        # PREVIEW SVM RESULTS  ##################################################################################
        print(f'\nb =', self.b)
        print('\nSUPPORT INFO')
        SUPPORT_INFO = p.DataFrame(
                                    {self.TARGET_HEADER[0][0]: self.SUPPORT_TARGETS,
                                    'ALPHAS': self.SUPPORT_ALPHAS}
        )
        print(SUPPORT_INFO)

        print(f'\nPREVIEW OF RESULTS')

        rows = len(self.ALPHAS) #50
        DISPLAY = p.DataFrame(
            {self.TARGET_HEADER[0][0]: self.TARGET_VECTOR[0][:rows], 'OUTPUT': self.OUTPUT_VECTOR[0][:rows],
             'CLASS': lf.link_fxns(self.OUTPUT_VECTOR[0], 'SVM_Perceptron')[:rows], 'ALPHAS': self.ALPHAS[:rows]}
        )
        print(DISPLAY)


    def print(self):
        try:
            # Build printout for specific ML module by piecing together
            self.iteration_print()
            self.ml_specific_print()
            self.conditional_print_for_all()
            self.always_printed_for_all()
        except:
            print(f'\nException trying to print results.  Perhaps objects havent been build yet.\n')


# INHERITED BY ALL ##################################################################################################
# custom_write                  # custom module for simplifying openpyxl data writing
# return_fxn()                  # Return
# conditional_dump_for_all()    # Define what to report based on what the output will be as determined by the final link function
# always_dumped_for_all()       # Statistics dumped for all ML packages
# OVERWRITTEN IN CHILD ##############################################################################################
# iteration_dump()              # Available for use by ML packages that have iterations
# ml_specific_dump()            # Dump instructions for specific ML module
# rsq()                         # Define function for rsq
# dump()                        # Build dump for specific ML module by piecing together


class SVMSummaryStatisticsDump(sst.SummaryStatisticsDumpTemplate):

    def __init__(self, wb, sheet_name, OUTPUT_VECTOR, TARGET, TARGET_HEADER, ALPHAS, SUPPORT_VECTORS,
                 SUPPORT_TARGETS, SUPPORT_ALPHAS, b, error, passes, max_passes):
        self.wb = wb
        self.TARGET_HEADER = TARGET_HEADER
        self.ALPHAS = ALPHAS
        self.SUPPORT_VECTORS = SUPPORT_VECTORS
        self.SUPPORT_TARGETS = SUPPORT_TARGETS
        self.SUPPORT_ALPHAS = SUPPORT_ALPHAS
        self.b = b
        self.passes = passes
        self.max_passes = max_passes

        self.link = 'SVM_Perceptron'

        self.row_counter = 0

        # DIFFERENT SHEET NAME FOR IF TRAIN, DEV, OR TEST, SPECIFIED INSITU
        # super().__init__(wb, OUTPUT_VECTOR, TARGET_VECTOR, link, error, sheet_name)
        super().__init__(self.wb, OUTPUT_VECTOR, TARGET, self.link, error, sheet_name)


    def iteration_dump(self):
        # Available for use by ML packages that have iterations
        # OVERWRITTEN IN CHILD

        self.row_counter += 2

        self.custom_write(self.sheet_name, self.row_counter, 2,
                          f'\nAfter pass {self.passes} of {self.max_passes}:', 'left', 'center', False)


    def ml_specific_dump(self):
        # Print instructions for specific ML module
        # OVERWRITTEN IN CHILD

        self.row_counter += 2

        # CALCULATE SVM STATISTICS #####################################################################

        self.custom_write(self.sheet_name, self.row_counter, 2, f'b', 'left', 'center', True)
        self.custom_write(self.sheet_name, self.row_counter, 3, self.b, 'center', 'center', False)

        self.row_counter += 2

        self.custom_write(self.sheet_name, self.row_counter, 2, 'SUPPORT INFO', 'left', 'center', True)

        self.row_counter += 1

        self.custom_write(self.sheet_name, self.row_counter, 2, self.TARGET_HEADER[0][0], 'center', 'center', True)
        self.custom_write(self.sheet_name, self.row_counter, 3, 'ALPHAS', 'center', 'center', True)
        for idx in range(len(self.SUPPORT_TARGETS)):
            self.row_counter += 1
            self.custom_write(self.sheet_name, self.row_counter, 2, self.SUPPORT_TARGETS[idx], 'center', 'center', False)
            self.custom_write(self.sheet_name, self.row_counter, 3, self.SUPPORT_ALPHAS[idx], 'center', 'center', False)

        self.row_counter += 2

        self.custom_write(self.sheet_name, self.row_counter, 2, f'PREVIEW OF RESULTS', 'center', 'center', True)

        self.row_counter += 1

        self.custom_write(self.sheet_name, self.row_counter, 2, self.TARGET_HEADER[0][0], 'center', 'center', True)
        self.custom_write(self.sheet_name, self.row_counter, 3, 'OUTPUT', 'center', 'center', True)
        self.custom_write(self.sheet_name, self.row_counter, 4, 'CLASS', 'center', 'center', True)
        self.custom_write(self.sheet_name, self.row_counter, 5, 'ALPHAS', 'center', 'center', True)

        rows = len(self.ALPHAS)  #50
        _ = lf.link_fxns(self.OUTPUT_VECTOR[0], 'SVM_Perceptron')
        for idx in range(rows):
            self.row_counter += 1
            self.custom_write(self.sheet_name, self.row_counter, 2, self.TARGET_VECTOR[0][idx], 'center', 'center', False)
            self.custom_write(self.sheet_name, self.row_counter, 3, self.OUTPUT_VECTOR[0][idx], 'center', 'center', False)
            self.custom_write(self.sheet_name, self.row_counter, 4, _[idx], 'center', 'center', False)
            self.custom_write(self.sheet_name, self.row_counter, 5, self.ALPHAS[idx], 'center', 'center', False)

        self.row_counter += 2

        self.custom_write(self.sheet_name, self.row_counter, 2, 'RSQ', 'left', 'center', False)
        self.custom_write(self.sheet_name, self.row_counter, 3, self.rsq(), 'center', 'center', False)

        self.row_counter += 1

        self.custom_write(self.sheet_name,
                          self.row_counter,
                          2,
                          f'\nMIN, MAX, AVG OUTPUT = {n.min(self.OUTPUT_VECTOR):.8f}, {n.max(self.OUTPUT_VECTOR):.8f}, {n.average(self.OUTPUT_VECTOR):.8f}',
                          'left',
                          'center',
                          False)


    def rsq(self):
        return n.corrcoef(self.TARGET_VECTOR[0].tolist(), lf.link_fxns(self.OUTPUT_VECTOR[0], 'SVM_Perceptron').tolist())[0][1]**2


    def dump(self):
        # Build excel dump for specific ML module by piecing together
        # OVERWRITTEN IN CHILD
        self.iteration_dump()
        self.ml_specific_dump()
        self.conditional_dump_for_all()
        # self.always_dumped_for_all()

        return self.return_fxn()




































