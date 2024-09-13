# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



"""

'visualize' is not implemented in GSTCVDask.

"""


# DASK
# ----
#
# Render the task graph for this parameter search using ``graphviz``.
#
# Requires ``graphviz`` to be installed.
#
# Parameters
# ----------
# filename : str or None, optional, default = None.
#    The name (without an extension) of the file to write to disk.
#    If `filename` is None, no file will be written, and we
#    communicate with dot using only pipes.
# format : {'png', 'pdf', 'dot', 'svg', 'jpeg', 'jpg'}, optional
#    Format in which to write output file.  Default is 'png'.
# **kwargs
#    Additional keyword arguments to forward to
#    ``dask.dot.to_graphviz``.
#
# Returns
# -------
# result : IPython.diplay.Image, IPython.display.SVG, or None
#    See ``dask.dot.dot_graph`` for more information.
#
# if hasattr(self, '_dask_estimator') and not self._dask_estimator:
#     raise NotImplementedError(f"Cannot visualize a sklearn estimator")
#
# self.check_is_fitted()
#
# return dask.visualize(
#     self._estimator,
#     filename=filename,
#     format=format,
#     **kwargs
# )
















































