# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import numpy as np
from pybear import new_numpy
from pybear.sparse_dict import _validation as val
from pybear.sparse_dict._transform import zip_array



def _create_random_sparse_dict(
    minimum:[int, float],
    maximum:[int, float],
    shape:tuple,
    sparsity:[int, float],
    dtype=float
) -> dict:

    """Create a random sparse dictionary with given min, max, shape, sparsity,
    and dtype. When dtype is an integer-type, discrete dense values are
    uniformly distributed over the interval [minimum, maximum-1]. Any value
    within the given interval is equally likely to be drawn. When dtype is a
    float-type, discrete dense values are uniformly distributed over the
    interval [minimum, maximum]. Again, any value within the given interval is
    equally likely to be drawn.

    Parameters
    ----------
    minimum:
        [int, float] - Lower boundary of the output interval. All values
        generated will be greater than or equal to minimum.
    maximum:
        [int, float] - Upper boundary of the output interval. For integer types,
        all values generated will be less than high; for float types, all
        values generated will be less than or equal to high.
    shape:
        tuple (int, int) - Output shape. Must be 1 or 2 dimensional. If the
        given shape is (m, n), then m * n samples are drawn.
    sparsity:
        [int, float] - the percentage of zeros in the output. Can also be
        interpreted as the likelihood that any given value in the output sparse
        dictionary is zero.
    dtype:
        numpy or python dtype, default = float. Dtype of the values in the
        returned sparse dictionary.


    Return
    ------
    SPARSE_DICT:
        dict - 'shape'-shaped sparse dictionary of random values from the
        distribution.

    See Also
    --------
    numpy.sparse_dict.random.randint
    numpy.sparse_dict.random.uniform

    Notes
    -----
    There is substantial performance drop-off when using non-python dtypes as
    values in dictionaries. Although use of numpy dtypes is available, the
    recommended dtype is standard python 'int' (the default).

    Examples
    --------
    >>> import numpy as np
    >>> from pybear import sparse_dict
    >>> data = sparse_dict.random._create_random_sparse_dict(
    ...                                       0, 5, (3,3), 50, dtype=np.uint8)
    >>> data     #doctest:+SKIP
    # {0: {0: 3, 1: 2, 2: 0}, 1: {2: 4}, 2: {1: 1, 2: 0}}
    """

    # 11/23/22 MAPPING THE DENSE VALUES OF A SPARSE ARRAY TO A SPARSE
    # DICT IS UNDER ALL CIRCUMSTANCES FASTEST WHEN USING np.nonzero TO GET
    # POSNS AND VALUES, THEN USING dict(()).

    # SUMMARY OF OPERATIONS
    # 1) VALIDATE PARAMS
    # 2) MAKE SPARSE NDARRAY WITH pybear.new_numpy._random.sparse
    # 3) CONVERT TO A SPARSE DICT WITH pybear._transform.zip_array


    _minimum = minimum
    _maximum = maximum
    _shape = shape
    _sparsity = sparsity
    _dtype = dtype

    del minimum, maximum, shape, sparsity, dtype

    __ = str(_dtype).upper()


    # INPUT VALIDATION ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    def is_numeric(name, value):
        try:
            float(value)
            if isinstance(value, bool):
                raise Exception
        except:
            raise TypeError(f"'{name}' must be a number, and not a boolean")

    def allowed_numeric(name, value):
        if value in (np.nan, float('-inf'), float('inf')):
            raise ValueError(f"disallowed {name} '{value}'")


    is_numeric('minimum', _minimum)
    allowed_numeric('minimum', _minimum)

    is_numeric('maximum', _maximum)
    allowed_numeric('maximum', _maximum)

    if _minimum >= _maximum:
        raise ValueError(f"low >= high")


    err_msg = (f"'shape' must be an integer >= 0 or a 1 or 2 "
                    f"dimensional array-like containing such values")
    try:
        iter(_shape)
        if isinstance(_shape, (dict, str)):
            raise UnicodeError
        if len(_shape) < 1 or len(_shape) > 2:
            raise UnicodeError
    except UnicodeError:
        raise TypeError(err_msg)
    except TypeError:
        try:
            float(_shape)
            _shape = (_shape, )
        except:
            raise TypeError(err_msg) from None
    except Exception as e:
        raise Exception(f"shape validation failed for uncontrolled reason --- "
                        f"{e}") from None


    if len(_shape) == 1:
        _len_outer = None
        _len_inner = _shape[0]
        is_numeric('shape[0]', _shape[0])
        allowed_numeric('shape[0]', _shape[0])
        try:
            val._is_int(_len_inner)
        except:
            raise ValueError(err_msg)
        if _len_inner < 0:
            raise ValueError(f'shape dimensions must be >= 0  ({_shape})')
    elif len(_shape) == 2:
        _len_outer = _shape[0]
        is_numeric('shape[0]', _shape[0])
        allowed_numeric('shape[0]', _shape[0])
        try:
            val._is_int(_len_outer)
        except:
            raise ValueError(err_msg)
        _len_inner = _shape[1]
        is_numeric('shape[1]', _shape[1])
        allowed_numeric('shape[1]', _shape[1])
        try:
            val._is_int(_len_inner)
        except:
            raise ValueError(err_msg)
        if _len_inner < 0 or _len_outer < 0:
            raise ValueError(f'shape dimensions must be >= 0  ({_shape})')


    is_numeric('sparsity', _sparsity)
    allowed_numeric('sparsity', _sparsity)

    if _sparsity > 100 or _sparsity < 0:
        raise ValueError(f'sparsity ({_sparsity}) must be 0 to 100.')

    try:
        np.array([]).astype(_dtype)
    except:
        raise TypeError(f"'dtype' must be a valid python or numpy dtype")

    del is_numeric, allowed_numeric

    # END INPUT VALIDATION  ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    # #########################################################################
    # HANDLE EDGE CASE SHAPES AND RETURN FROM HERE ############################

    if len(_shape) == 1:

        if _shape[0] == 0:
            return {}
        else:  # _shape[0] must be > 0
            return zip_array(
                             new_numpy.random.sparse(_minimum,
                                                     _maximum,
                                                     _shape,
                                                     _sparsity,
                                                     _dtype),
                             _dtype
                             )

    elif len(_shape) == 2:
        if _shape == {0,0}:
            return {}
        elif _shape[0] == 0:
            return {}
        elif _shape[1] == 0:
            return {int(i): {} for i in range(_shape[0])}

    # END EDGE CASE SHAPES ####################################################
    # #########################################################################

    ###########################################################################
    # IF _sparsity==100, JUST BUILD HERE, BYPASSING EVERYTHING BELOW ##########
    if _sparsity == 100:
        if len(_shape) == 1:
            SPARSE_DICT = {int(_len_inner - 1): _dtype(0)}
        elif len(_shape) == 2:
            SPARSE_DICT = {int(i):
                   {int(_len_inner-1): _dtype(0)} for i in range(_len_outer)
            }

        return SPARSE_DICT
    # END IF _sparsity==100 ###################################################
    ###########################################################################

    ARRAY = new_numpy.random.sparse(_minimum, _maximum, _shape, _sparsity, _dtype)

    SPARSE_DICT = zip_array(ARRAY, _dtype)

    # END FULLY SIZED NDARRAY W MASK ##########################################
    ###########################################################################

    return SPARSE_DICT


def randint(minimum:int, maximum:int, shape:tuple, sparsity:float, dtype=int):

    """
    Return a sparse dictionary of given sparsity, shape, and dtype, with the
    dense values being random integers from the “discrete uniform” distribution
    in the interval [minimum, maximum-1].

    Parameters
    ----------
    minimum:
        int - the lowest integer to be drawn from the distribution
    maximum:
        int - one above the largest integer to be drawn from the
        distribution
    shape:
        tuple (int, int) - Output shape. Must be 1 or 2 dimensional. If the
        given shape is (m, n), then m * n samples are drawn.
    sparsity:
        [int, float] - the percentage of zeros in the output. Can also be
        interpreted as the likelihood that any given value in the output sparse
        dictionary is zero.
    dtype:
        numpy or python dtype, default = int. Dtype of the values in the
        returned sparse dictionary. Must be an integer dtype.

    Return
    ------
    SPARSE_DICT:
        dict - 'shape'-shaped sparse dictionary of random integers from the
        distribution.

    See Also
    --------
    numpy.random.randint

    Notes
    -----
    There is substantial performance drop-off when using non-python dtypes as
    values in dictionaries. Although use of numpy dtypes is available, the
    recommended dtype is standard python 'int' (the default).

    Examples
    --------
    >>> import numpy as np
    >>> from pybear import sparse_dict
    >>> data = sparse_dict.random.randint(0, 5, (20,), 80, dtype=np.uint8)
    >>> data   #doctest:+SKIP
    {2: 4, 4: 4, 14: 4, 16: 1, 19: 0}

    """

    val._is_int(minimum)

    val._is_int(maximum)

    if minimum >= maximum:
        raise ValueError(f"low >= high")

    if 'FLOAT' in str(dtype).upper():
        raise TypeError(f"Unsupported dtype {dtype} for randint")

    return _create_random_sparse_dict(minimum, maximum, shape, sparsity, dtype)


def uniform(minimum:float, maximum:float, shape:tuple, sparsity:float):

    """
    Return a sparse dictionary of given sparsity and shape with the
    dense values being randomly selected from the interval [minimum, maximum].

    Parameters
    ----------
    minimum:
        [int, float] - the lowest value to be drawn from the distribution
    maximum:
        [int, float] - the largest value to be drawn from the distribution
    shape:
        tuple (int, int) - Output shape. Must be 1 or 2 dimensional. If the
        given shape is (m, n), then m * n samples are drawn.
    sparsity:
        [int, float] - the percentage of zeros in the output. Can also be
        interpreted as the likelihood that any given value in the output sparse
        dictionary is zero.

    Return
    ------
    SPARSE_DICT:
        dict - 'shape'-shaped sparse dictionary of random values from the
        distribution.

    See Also
    --------
        numpy.random.randint

    Examples
    --------
    >>> import numpy as np
    >>> from pybear import sparse_dict
    >>> data = sparse_dict.random.uniform(0, 5, (10,), 80)
    >>> data   #doctest:+SKIP
    {3: 0.3473838764565701, 5: 4.981795571026054, 9: 0.0}

    """

    # min CAN BE >= max
    minimum, maximum = min(minimum, maximum), max(minimum, maximum)

    return _create_random_sparse_dict(minimum, maximum, shape, sparsity, float)











