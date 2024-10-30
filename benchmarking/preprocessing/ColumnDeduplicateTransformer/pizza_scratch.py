# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

import sys


# try:
#     raise AttributeError()
# except:
#     __ = sys.exc_info()
#     for idx, crap in enumerate(__):
#         print(f'{idx}) {crap}')



from sklearn.preprocessing import StandardScaler

ss = StandardScaler(with_mean=True, with_std=True)


ss.get_feature_names_out()




