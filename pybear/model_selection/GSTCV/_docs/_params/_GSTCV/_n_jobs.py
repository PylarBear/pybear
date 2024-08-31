# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


"""

    _n_jobs:
        Union[int, None], default=None - Number of jobs to run in
        parallel. -1 means using all processors.

        For best speed benefit, pybear recommends setting n_jobs in both
        GSTCV and the wrapped estimator to None, whether under a joblib
        context manager or standing alone. When under a joblib context
        manager, also set n_jobs in the context manager to None.

"""


# Notes
# -----
# this is a grid of run times for GSTCV with different combinations
# of GSTCV njobs and estimator njobs, with n_jobs being hard-set
# via n_jobs kwargs. the fastest combinations are Nones and
# 1s.  the slowest is GSTCV in [0,1] and estimator > 1
# GSTCV:
#         gstcv_None	gstcv_1	gstcv_2	gstcv_3	gstcv_4
# est_None	4.3	    4.1	    13.8	15.5	13.2
# est_1	    4	    4	    13.8	15.4	12.9
# est_2	    72.8	71.4	13.3	18.2	19.4
# est_3	    72.2	71.7	15.8	15.7	15.1
# est_4	    71.1	71.9	16.5	15.7	14.6
#
# sk GSCV:
#         gscv_None	gscv_1	gscv_2	gscv_3	gscv_4
# est_None	3.7	    3.4	    4.5 	4	    2.5
# est_1	    3.4	    3.5	    4.1	    3.9	    2.5
# est_2	    73.3	72.7	3.7	    5.8	    6.2
# est_3	    71.6	70.8	6.1	    3.9	    3.8
# est_4	    71	    70.7	6.4	    3.9	    3.2
#
# with context manager and GS(T)CV & est kwargs set to None:
# GSTCV:
# gstcv_time
# CM_None	4.1
# CM_1	3.7
# CM_2	4
# CM_3	4.1
# CM_4	4
#
# GSCV:
# gscv_time
# CM_None	3.3
# CM_1	3.3
# CM_2	7.1
# CM_3	7.5
# CM_4	7.7









