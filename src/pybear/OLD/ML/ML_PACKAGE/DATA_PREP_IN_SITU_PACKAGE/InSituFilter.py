from ML_PACKAGE.DATA_PREP_PRE_RUN_PACKAGE import PreRunFilter as prf


class InSituFilter(prf.PreRunFilter):
    def __init__(self,
                 standard_config,
                 user_manual_or_standard,
                 filter_method,
                 SUPER_RAW_NUMPY_LIST,
                 data_given_orientation,
                 target_given_orientation,
                 refvecs_given_orientation,
                 FULL_SUPOBJS,
                 CONTEXT,
                 KEEP,
                 SUPER_RAW_NUMPY_LIST_BASE_BACKUP,
                 FULL_SUPOBJS_BASE_BACKUP,
                 CONTEXT_BASE_BACKUP,
                 KEEP_BASE_BACKUP,
                 SUPER_RAW_NUMPY_LIST_GLOBAL_BACKUP,
                 FULL_SUPOBJS_GLOBAL_BACKUP,
                 KEEP_GLOBAL_BACKUP,
                 bypass_validation
                ):

        # 12-12-21 NO GLOBAL BACKUPS FOR VDATATYPES, MDATATYPES, FILTERING, MIN_CUTOFF, USE_OTHER, THESE ARE ALL
        # EMPTY AT GLOBAL SNAPSHOT POINT

        super().__init__(standard_config, user_manual_or_standard, filter_method, SUPER_RAW_NUMPY_LIST, data_given_orientation,
                 target_given_orientation, refvecs_given_orientation, FULL_SUPOBJS, CONTEXT, KEEP, SUPER_RAW_NUMPY_LIST_BASE_BACKUP,
                 FULL_SUPOBJS_BASE_BACKUP, CONTEXT_BASE_BACKUP, KEEP_BASE_BACKUP, SUPER_RAW_NUMPY_LIST_GLOBAL_BACKUP,
                 FULL_SUPOBJS_GLOBAL_BACKUP, KEEP_GLOBAL_BACKUP, bypass_validation)


    # WHOLE LOT OF INHERITED IN HERE


    def mode(self):
        return('INSITU')


    def return_fxn(self):
        return self.return_fxn()[:4]


########################################################################################################################
########################################################################################################################
########################################################################################################################




if __name__ == '__main__':
    pass
    # InSituFilter(
    #
    # )





