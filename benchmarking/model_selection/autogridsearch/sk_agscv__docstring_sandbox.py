# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


# see if different configurations of the wrapper and __doc__ assignments
# in SkLearnAutoGridSearch and DaskAutoGridSearch expose the docs in
# autogridsearch_docs to the pycharm tool tip. 24_06_03_14_01_00 still no :(

# pizza, need to see under what arrangement sphinx can see the docs.




from pybear.model_selection import AutoGridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.datasets import make_classification

import numpy as np



# hover over to see if docs are visible in tool tip

gscv = AutoGridSearchCV(
    estimator=LogisticRegression(),
    params={'C':[np.logspace(0, 6, 7),[7, 11, 11],'soft_float']}
)

X, y = make_classification(n_features=5, n_samples=50)








