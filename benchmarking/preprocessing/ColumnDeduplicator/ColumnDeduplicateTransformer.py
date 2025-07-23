# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.preprocessing import ColumnDeduplicator


import numpy as np
import pandas as pd
from uuid import uuid4



CDT = ColumnDeduplicator()

_shape = (20,10)
data = np.random.uniform(0,1,_shape)
_columns1 = [str(uuid4())[:4] for _ in range(_shape[1])]
_columns2 = [str(uuid4())[:4] for _ in range(_shape[1])]


DF1 = pd.DataFrame(data=data, columns=_columns1)
DF2 = pd.DataFrame(data=data, columns=_columns2)
DFNone = pd.DataFrame(data=data, columns=None)

CDT.partial_fit(DFNone)
CDT.partial_fit(DF2)
CDT.transform(DF1)




