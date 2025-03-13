import numpy as n
from ML_PACKAGE.GENERIC_PRINT import SummaryStatisticsTemplate as sst


# INHERITED BY ALL ##################################################################################################
# rsq()                         # Define function for rsq
# conditional_print_for_all()   # Define what to report based on what the output will be as determined by the final link function
# always_printed_for_all()      # Statistics printed for all ML packages
# OVERWRITTEN IN CHILD ##############################################################################################
# iteration_print()             # Available for use by ML packages that have iterations
# ml_specific_print()           # Print instructions for specific ML module
# print()                       # Build printout for specific ML module by piecing together


class NNSummaryStatisticsPrint(sst.SummaryStatisticsPrintTemplate):

    def __init__(self, OUTPUT_VECTOR, TARGET_VECTOR, link, error, ARRAY_OF_NODES, iteration, gd_iterations):
        self.ARRAY_OF_NODES = ARRAY_OF_NODES
        self.iteration = iteration
        self.gd_iterations = gd_iterations

        super().__init__(OUTPUT_VECTOR, TARGET_VECTOR, link, error)


    def iteration_print(self):
        # Available for use by ML packages that have iterations
        print(f'\nAfter iteration {self.iteration} of {self.gd_iterations}:')
        pass


    def ml_specific_print(self):
        # Print instructions for specific ML module

        # CALCULATE NN ELEMENT STATISTICS #####################################################################
        max_matrix_element = float('-inf')
        min_matrix_element = float('inf')
        count_of_elements = 0
        sum_of_elements = 0
        for ARRAY in self.ARRAY_OF_NODES:
            if n.max(ARRAY) > max_matrix_element: max_matrix_element = n.max(ARRAY)
            if n.min(ARRAY) < min_matrix_element: min_matrix_element = n.min(ARRAY)
            sum_of_elements += n.sum(ARRAY)
            for LIST in ARRAY:
                count_of_elements += len(LIST)
        mean_of_elements = sum_of_elements / count_of_elements

        print(f'\nMIN, MAX, AVG NN_ELEMENT = {min_matrix_element:.8f}, {max_matrix_element:.8f}, {mean_of_elements:.8f}')


    def print(self):
        # Build printout for specific ML module by piecing together
        self.iteration_print()
        self.ml_specific_print()
        self.conditional_print_for_all()
        self.always_printed_for_all()


# INHERITED BY ALL ##################################################################################################
# custom_write                  # custom module for simplifying openpyxl data writing
# return_fxn()                  # Return
# rsq()                         # Define function for rsq
# conditional_dump_for_all()    # Define what to report based on what the output will be as determined by the final link function
# always_dumped_for_all()       # Statistics dumped for all ML packages
# OVERWRITTEN IN CHILD ##############################################################################################
# iteration_dump()              # Available for use by ML packages that have iterations
# ml_specific_dump()            # Dump instructions for specific ML module
# dump()                        # Build dump for specific ML module by piecing together



class NNSummaryStatisticsDump(sst.SummaryStatisticsDumpTemplate):

    def __init__(self, wb, OUTPUT_VECTOR, TARGET_VECTOR, link, error, sheet_name, ARRAY_OF_NODES, iteration, gd_iterations):
        self.ARRAY_OF_NODES = ARRAY_OF_NODES
        self.iteration = iteration
        self.gd_iterations = gd_iterations

        # DIFFERENT SHEET NAME FOR IF TRAIN, DEV, OR TEST, SPECIFIED INSITU
        super().__init__(wb, OUTPUT_VECTOR, TARGET_VECTOR, link, error, sheet_name)



    def iteration_dump(self):
        # Available for use by ML packages that have iterations
        # OVERWRITTEN IN CHILD

        self.row_counter += 2

        self.custom_write(self.sheet_name, self.row_counter, 2,
                          f'\nAfter iteration {self.iteration} of {self.gd_iterations}:', 'left', 'center', False)


    def ml_specific_dump(self):
        # Print instructions for specific ML module
        # OVERWRITTEN IN CHILD

        self.row_counter += 2

        # CALCULATE NN ELEMENT STATISTICS #####################################################################
        max_matrix_element = float('-inf')
        min_matrix_element = float('inf')
        count_of_elements = 0
        sum_of_elements = 0
        for ARRAY in self.ARRAY_OF_NODES:
            if n.max(ARRAY) > max_matrix_element: max_matrix_element = n.max(ARRAY)
            if n.min(ARRAY) < min_matrix_element: min_matrix_element = n.min(ARRAY)
            sum_of_elements += n.sum(ARRAY)
            for LIST in ARRAY:
                count_of_elements += len(LIST)
        mean_of_elements = sum_of_elements / count_of_elements

        self.custom_write(self.sheet_name, self.row_counter, 2, f'\nMIN, MAX, AVG NN_ELEMENT', 'left', 'center', False)
        self.custom_write(self.sheet_name,
                          self.row_counter,
                          2 + 4,
                          f'{min_matrix_element:.8f}, {max_matrix_element:.8f}, {mean_of_elements:.8f}',
                          'left',
                          'center',
                          False)


    def dump(self):
        # Build excel dump for specific ML module by piecing together
        # OVERWRITTEN IN CHILD
        self.iteration_dump()
        self.ml_specific_dump()
        self.conditional_dump_for_all()
        self.always_dumped_for_all()

        return self.return_fxn()



