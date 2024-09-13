# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



"""

Set the parameters of the GSTCV(Dask) instance or the embedded
estimator. The method works on simple estimators as well as on
nested objects (such as Pipeline). The parameters of single
estimators can be updated using 'estimator__<parameter>'.
Pipeline parameters can be updated using the form
'estimator__<pipe_parameter>. Steps of a pipeline have parameters
of the form <step>__<parameter> so that it’s also possible to
update a step's parameters. The parameters of steps in the
pipeline can be updated using 'estimator__<step>__<parameter>'.


Parameters
----------
**params:
    dict[str: any] - GSTCV(Dask) and/or estimator parameters.


Return
------
-
    self: estimator instance - GSTCV(Dask) instance.

"""




