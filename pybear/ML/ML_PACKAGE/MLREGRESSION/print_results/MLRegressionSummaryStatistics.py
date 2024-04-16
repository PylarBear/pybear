from ML_PACKAGE.GENERIC_PRINT import SummaryStatisticsTemplate as sst


# INHERITED BY ALL ##################################################################################################
# rsq()                         # Define function for rsq
# conditional_print_for_all()   # Define what to report based on what the output will be as determined by the final link function
# always_printed_for_all()      # Statistics printed for all ML packages
# OVERWRITTEN IN CHILD ##############################################################################################
# iteration_print()             # Available for use by ML packages that have iterations
# ml_specific_print()           # Print instructions for specific ML module
# print()                       # Build printout for specific ML module by piecing together



class MLRegressionSummaryStatisticsPrint(sst.SummaryStatisticsPrintTemplate):

    def __init__(self, OUTPUT_VECTOR, TARGET_VECTOR, link, error):

        super().__init__(OUTPUT_VECTOR, TARGET_VECTOR, link, error)


    def ml_specific_print(self):
        # Print instructions for specific ML module
        # OVERWRITTEN IN CHILD
        pass


    def print(self):
        # Build printout for specific ML module by piecing together
        self.conditional_print_for_all()
        self.always_printed_for_all()
        self.ml_specific_print()


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



class MLRegressionSummaryStatisticsDump(sst.SummaryStatisticsDumpTemplate):

    def __init__(self, wb, OUTPUT_VECTOR, TARGET_VECTOR, link, error, sheet_name):

        # DIFFERENT SHEET NAME FOR IF TRAIN, DEV, OR TEST, SPECIFIED INSITU
        super().__init__(wb, OUTPUT_VECTOR, TARGET_VECTOR, link, error, sheet_name)


    def ml_specific_dump(self):
        # Print instructions for specific ML module
        # OVERWRITTEN IN CHILD

        # self.row_counter += 2

        pass


    def dump(self):
        # Build excel dump for specific ML module by piecing together
        # OVERWRITTEN IN CHILD
        self.conditional_dump_for_all()
        self.always_dumped_for_all()
        self.ml_specific_dump()

        return self.return_fxn()















