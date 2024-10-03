# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


# pizza decide what to do here



# header handling - - - - -
if isinstance(X, pd.core.frame.DataFrame):
    X = pd.DataFrame(X, columns=HEADER)
elif isinstance(X, np.ndarray):
    pass
# END header handling - - - - -



