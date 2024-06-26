# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import numpy as np
from feature_extraction.text._StringToToken.StringToToken import StringToToken
import dask.array as da
from dask_ml.wrappers import ParallelPostFit


chars = 'abcdefghijklmnopqrstuvwxyz     '

X = []
for _ in range(100):
    rand_total_len = np.random.randint(50, 100)
    rand_chars = np.random.choice(list(chars),rand_total_len, replace=True)
    rand_string = "".join(rand_chars)
    X.append(rand_string)

X = da.array(np.array(X,dtype='<U10000')).reshape((1,-1)).rechunk((1,10))

print(f'\nIN X:')
print(X.compute())

stt = StringToToken(sep=None, maxsplit=-1, pad='')


wrapped_stt = ParallelPostFit(stt)

OUT_X = wrapped_stt.transform(X)

print(f'\nOUT_X:')
print(OUT_X.compute())








