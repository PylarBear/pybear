# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import numpy as np

from pybear.preprocessing._SlimPolyFeatures.SlimPolyFeatures import \
    SlimPolyFeatures as SPF



X = np.random.randint(0, 10, (10, 5))


trfm = SPF(
    degree = 3,
    min_degree = 1,
    interaction_only = False,
    scan_X = True,
    keep = 'first',
    sparse_output = False,
    feature_name_combiner = 'as_feature_names',
    equal_nan = True,
    rtol = 1e-05,
    atol = 1e-08,
    n_jobs = -1
)

trfm.fit_transform(X)








