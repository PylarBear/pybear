# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import numpy as np
import pandas as pd
import scipy.sparse as ss

from typing_extensions import Union, TypeAlias
import numpy.typing as npt



SparseTypes: TypeAlias = Union[
    ss._csr.csr_matrix,
    ss._csc.csc_matrix,
    ss._coo.coo_matrix,
    ss._dia.dia_matrix,
    ss._bsr.bsr_matrix,
    ss._csr.csr_array,
    ss._csc.csc_array,
    ss._coo.coo_array,
    ss._dia.dia_array,
    ss._bsr.bsr_array
]




def inf_mask(
    obj: Union[npt.NDArray, pd.DataFrame, SparseTypes]
) -> npt.NDArray[bool]:

    """
    Return a boolean numpy array or vector indicating the locations of
    infinity-like values in the data. "Infinity-like values" include, at
    least, numpy.inf, -numpy.inf, numpy.PINF, numpy.NINF, math.inf,
    -math.inf, str('inf'), str('-inf'), float('inf'), 'float('-inf'),
    'decimal.Decimal('Infinity'), and 'decimal.Decimal('-Infinity').

    This module accepts numpy arrays, pandas dataframes, and all scipy
    sparse matrices/arrays except dok and lil formats. In all cases,
    the boolean mask is generated from a numpy array representation of
    the data. For numpy arrays, they are handled as is. For pandas
    dataframes, the given data is converted to a numpy array via the
    to_numpy() method for handling. For scipy sparse objects, the 'data'
    attribute (which is a numpy ndarray) is extracted.

    In the cases of numpy arrays and pandas dataframes of shape
    (n_samples, n_features), return an identically shaped boolean numpy
    array. In the case of a numpy 1D vector, return an identically shaped
    boolean numpy vector. In the cases of scipy sparse objects, return a
    boolean numpy vector of shape equal to that of the 'data' attribute
    of the sparse object.

    'dok' is the only scipy sparse format that doesnt have a 'data'
    attribute, and for that reason it is not handled by inf_mask().
    scipy sparse 'lil' cannot be masked in an elegant way, and for that
    reason it is also not handled by inf_mask(). All other scipy sparse
    formats only take numeric data.

    This module relies heavily on numpy.isinf to locate infinity-like
    values in float dtype data. All infinity-like forms mentioned above
    are found by this function in float dtype data.

    Of the containers handled by this module, none of them allow for
    infinity-like values in integer dtype data. This makes for
    straightforward handling of these objects, in that every position in
    the returned boolean mask must be False.

    String and object dtype data are not handled by the numpy.isinf
    function. Fortunately, at creation of a string dtype numpy array,
    if there are float or string infinity-like values in it almost all
    of them are coerced to str('inf') or str('-inf'). The exception is
    decimal.Decimal('Infinity') and decimal.Decimal('-Infinity'), which
    are coerced to str('Infinity') and str('-Infinity'). Building a mask
    from this is straightforward. But object dtype numpy arrays do not
    make these conversions, so the float infinity-likes stay in the
    object array in float format. This poses a problem because
    numpy.isinf cannot take object formats, but it is very plausible
    that there are infinity-likes in it. So object dtype data are to
    cast to string dtype, which forces the conversion.


    Parameters
    ----------
    obj:
        NDArray, pandas.DataFrame, or scipy.sparse[float, int], of shape
        (n_samples, n_features) or (n_samples, ) - the object for which
        to mask infinity-like representations.


    Return
    ------
    mask:
        NDArray[bool], of shape (n_samples, n_features) or of shape
        (n_samples, ) or of shape (n_non_zero_values, ), indicating
        infinity-like representations in 'obj' via the value boolean
        True. Values that are not infinity-like are False.

    """

    _err_msg = (
        f"'obj' must be an array-like with a copy() method, such as numpy "
        f"array, pandas dataframe, or scipy sparse matrix or array. if "
        f"passing a scipy sparse object, it cannot be dok or lil. python "
        f"builtin iterables such as list, tuple, or set, are not allowed."
    )

    try:
        iter(obj)
        if isinstance(obj, (str, dict, list, tuple, set)):
            raise Exception
        if not hasattr(obj, 'copy'):
            raise Exception
        if hasattr(obj, 'toarray'):
            if not hasattr(obj, 'data'):  # ss dok
                raise Exception
            elif all(map(isinstance, obj.data, (list for _ in obj.data))):  # ss lil
                raise Exception
            else:
                obj = obj.data
    except:
        raise TypeError(_err_msg)

    _ = obj.copy()

    try:
        _ = _.to_numpy()
    except:
        pass


    # want to be able to handle int dtype objects. if obj is int dtype,
    # then it cant possibly have 'inf' int it. to avoid converting an int
    # dtype over to float64 (even if it would be only a transient state),
    # look to see if it is int dtype and just return a mask of Falses.
    if 'int' in str(_.dtype).lower():
        return np.zeros(_.shape).astype(bool)

    try:
        # np.isinf cannot take non-num dtype. try to coerce the data to
        # float64, if it wont go, try to handle it as string/object.
        # otherwise, if data is already float dtype, then we are good here.
        return np.isinf(_.astype(np.float64)).astype(bool)
    except:
        # fortunately, at creation of an str np array, if there are float
        # or str inf-likes in it, almost all of them are coerced to
        # str('inf') or str('-inf'). the exception is decimal.Decimal('Infinity')
        # and decimal.Decimal('-Infinity'), which are coerced to str('Infinity')
        # and str('-Infinity'). so this is elegant enough to handle. but
        # object dtype np arrays do not make these coersions, the float
        # inf-likes stay in the object array in float format. this poses
        # a problem because np.isinf cannot take object formats, but it
        # is very plausible that there are inf-likes in it. so need to
        # convert the object dtype to str, which will force the coersion.
        _ = _.astype(str)
        _ = (_ == 'inf') + (_ == '-inf') + (_ == 'Infinity') + (_ == '-Infinity')
        return _.astype(bool)














