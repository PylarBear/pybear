# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



"""

    iid:
        boolean, default=True - iid is ignored when cv is an iterable.
        Indicates whether the data's examples are believed to have random
        distribution (True) or if the examples are organized non-randomly
        in some way (False). If the data is not iid, dask KFold will
        cross chunk boundaries when reading the data in an attempt to
        randomize the data; this can be an expensive process. Otherwise,
        if the data is iid, dask KFold can handle the data as chunks
        which is much more efficient.


"""






