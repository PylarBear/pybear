# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

import numpy as np
import pandas as pd
import pybear.sparse_dict._validation as val
from dask import array as da
from dask import dataframe as ddf
from pybear.sparse_dict.sparse_dict import core_merge_outer
from pybear.sparse_dict._linalg import core_sparse_transpose
from pybear.sparse_dict._utils import shape_, inner_len, clean


def zip_array(ARRAY:[list, tuple, set, np.ndarray], dtype=float) -> dict:
    """
    Convert an array-like to a sparse dictionary. Must be a 1 or 2 dimensional
    array-like. Accepts ragged arrays, but does not correct the raggedness.


    Parameters
    ----------
    ARRAY:
        [list, set, tuple, np.ndarray] - The object to convert to a sparse
        dictionary.
    dtype:
        [int, float, np.int8, np.int16, np.int32, np.int64, np.uint8,
        np.uint16, np.uint32, np.uint64, np.float16, np.float32, np.float64],
        default = float - the python or numpy dtype of the values in the
        returned sparse dictionary. See Notes. Dictionary keys are always
        python integers and cannot be changed.

    Return
    ------
    SPARSE_DICT:
        dict - For an array-like object with 2-dimensional shape, e.g.,
        (1,2), an outer dictionary is returned - {0: {0:1, 1:2}}.
        For an array-like with 1-dimensional shape, e.g., (3, ), an inner
        dictionary is returned - {0:1, 1:4, 2:3}

    Notes
    -----
    There is substantial performance drop-off when using non-python dtypes as
    values in dictionaries. Although use of numpy dtypes is available, the
    recommended dtypes are standard python 'int' and 'float'.

    Examples
    --------
    >>> from pybear.sparse_dict import zip_array
    >>> data = [[0,6,3],[2,4,0],[7,4,1]]
    >>> OUTPUT = zip_array(data, dtype=int)
    >>> print(OUTPUT)
    {0: {1: 6, 2: 3}, 1: {0: 2, 1: 4, 2: 0}, 2: {0: 7, 1: 4, 2: 1}}

    """


    val._insufficient_list_args_1(ARRAY)

    __ = str(type(ARRAY)).upper()
    if "DATAFRAME" in __ or "SERIES" in __:
        raise TypeError(f"'ARRAY' cannot be a DataFrame or Series")

    if "DASK" in __:
        raise TypeError(f"'ARRAY' cannot be a dask object. Use zip_dask_array.")

    del __

    _dtype = dtype

    is_inner = False  # set later

    try:
        # will TypeError if is []
        list(map(iter, ARRAY))
        # if passes it is [[]]
        try:
            # will ValueError if is ragged
            np.array(ARRAY)
        except ValueError:
            # leave it as ragged lists
            pass
        except Exception as e:
            raise AssertionError(f'Raggedness test failed for reason other than '
                     f'ValueErrror --- {e}')
    except TypeError:
        # if is [], reshape to [[]], and note that it was an inner

        if isinstance(ARRAY, set):
            ARRAY = list(ARRAY)

        ARRAY = np.array(ARRAY, dtype=_dtype).reshape((1,-1))

        is_inner = True

    except AssertionError:
        raise

    except ValueError:
        pass

    except Exception as e:
        raise AssertionError(f'iterable test failed for reason other than '
                             f'ValueError --- {e}')


    if isinstance(ARRAY, np.ndarray):

        if ARRAY.shape in ((0,0), (0,1)):
            return {}
        elif ARRAY.shape == (1,0):
            return {0:{}}
        elif len(ARRAY.shape) > 2:
            raise ValueError(f"'ARRAY' must be 1 or 2 dimensional")

    else:
        # all of this just to handle lists
        try:
            list(map(iter, ARRAY))

            if ARRAY == [[]]:
                return {0:{}}

            # if this passes, is more than 2D
            ctr = 0
            for _ in ARRAY:
                try:
                    list(map(iter, _))
                    ctr += 1
                except TypeError:
                    pass

            if ctr != 0:
                raise ValueError(f"'ARRAY' must be 1 or 2 dimensional")

        except TypeError:
            pass
        except ValueError:
            raise
        except Exception as e:
            raise Exception(f"list handling excepted for uncontrolled reason "
                            f"--- {e}")


    SPARSE_DICT = {}

    if len(ARRAY) == 0:
        return SPARSE_DICT

    for outer_key in range(len(ARRAY)):

        # 22_11_23 MAPPING THE DENSE VALUES OF A SPARSE ARRAY TO A SPARSE
        # DICT IS UNDER ALL CIRCUMSTANCES FASTEST WHEN USING np.nonzero TO GET
        # POSNS AND VALUES, THEN USING dict(()) TO MAP.

        # OBSERVE PLACEHOLDER RULES, ALWAYS LAST INNER IDX INCLUDED
        NON_ZERO_KEYS = np.nonzero(ARRAY[outer_key][:-1])[0]

        # Create posn for last value even if value is zero, so that original
        # length of the object is retained.
        if len(ARRAY[outer_key])-1 not in NON_ZERO_KEYS:
            NON_ZERO_KEYS = np.hstack((
                                        NON_ZERO_KEYS,
                                        len(ARRAY[outer_key])-1
            )).astype(int)

        NON_ZERO_VALUES = np.array(ARRAY[outer_key])[NON_ZERO_KEYS]

        if _dtype in (int, float):
            NON_ZERO_VALUES = list(map(_dtype, NON_ZERO_VALUES.tolist()))
        else:
            NON_ZERO_VALUES = NON_ZERO_VALUES.astype(_dtype)


        SPARSE_DICT[int(outer_key)] = dict((
                            zip(NON_ZERO_KEYS.tolist(),NON_ZERO_VALUES)
        ))

    del NON_ZERO_KEYS, NON_ZERO_VALUES

    if is_inner:
        SPARSE_DICT = SPARSE_DICT[0]

    return SPARSE_DICT



def zip_dask_array(ARRAY:da.core.Array, dtype=float) -> dict:
    """
    Convert a dask array to a sparse dictionary. Must be a 1 or 2 dimensional
    dask array only.

    Parameters
    ----------
    ARRAY:
        [dask.array.core.Array] - The dask array to convert to a sparse
        dictionary.
    dtype:
        [int, float, np.int8, np.int16, np.int32, np.int64, np.uint8,
        np.uint16, np.uint32, np.uint64, np.float16, np.float32, np.float64],
        default = float - the python or numpy dtype of the values in the
        returned sparse dictionary. See Notes. Dictionary keys are always
        python integers and cannot be changed.

    Return
    ------
    SPARSE_DICT:
        dict - For a dask array with 2-dimensional shape, e.g., (1,2),
        an outer dictionary is returned - {0: {0:1, 1:2}}. For a dask array
        with 1-dimensional shape, e.g., (3, ), an inner dictionary is
        returned - {0:1, 1:4, 2:3}

    See Also
    --------
    pybear.sparse_dict.zip_array
    dask.array.core.Array

    Notes
    -----
    There is substantial performance drop-off when using non-python dtypes as
    values in dictionaries. Although use of numpy dtypes is available, the
    recommended dtypes are standard python 'int' and 'float'.

    Examples
    --------
    >>> from pybear.sparse_dict import zip_dask_array
    >>> from dask import array as da
    >>> data = da.array([[0,1,2],[1,0,1],[2,1,0],[0,0,0]])
    >>> data = data.rechunk((2,3))
    >>> OUTPUT = zip_dask_array(data, dtype=int)
    >>> print(OUTPUT)
    {0: {1: 1, 2: 2}, 1: {0: 1, 2: 1}, 2: {0: 2, 1: 1, 2: 0}, 3: {2: 0}}

    """

    if not isinstance(ARRAY, da.core.Array):
        raise TypeError(f'only accepts dask arrays')

    _dtype = dtype

    is_inner = False  # set later

    if ARRAY.shape in ((0,), (0, 0), (0, 1)):
        return {}
    elif ARRAY.shape == (1, 0):
        return {0: {}}
    elif len(ARRAY.shape) == 1:
        ARRAY = ARRAY.reshape((1,-1))
        is_inner = True
    elif len(ARRAY.shape) == 2:
        pass
    elif len(ARRAY.shape) > 2:
        raise ValueError(f"'ARRAY' must be 1 or 2 dimensional")
    else:
        raise Assertionerror(f"Unable to get shape for 'ARRAY'")

    def core_inner_dict_builder(ARRAY):

        nonlocal _dtype

        _SPARSE_DICT = {}

        for outer_key in range(len(ARRAY)):

            # OBSERVE PLACEHOLDER RULES, ALWAYS LAST INNER IDX INCLUDED
            NON_ZERO_KEYS = np.nonzero(ARRAY[outer_key][:-1])[0]

            # Create posn for last value even if value is zero, so that original
            # length of the object is retained.
            if len(ARRAY[outer_key])-1 not in NON_ZERO_KEYS:
                NON_ZERO_KEYS = np.hstack((
                                            NON_ZERO_KEYS,
                                            len(ARRAY[outer_key])-1
                )).astype(int)

            NON_ZERO_VALUES =  np.array(ARRAY[outer_key])[NON_ZERO_KEYS]

            if _dtype in (int, float):
                NON_ZERO_VALUES = list(map(_dtype, NON_ZERO_VALUES.tolist()))
            else:
                NON_ZERO_VALUES = NON_ZERO_VALUES.astype(_dtype)

            _SPARSE_DICT[int(outer_key)] = dict((
                                    zip(NON_ZERO_KEYS.tolist(), NON_ZERO_VALUES)
            ))

            del NON_ZERO_KEYS, NON_ZERO_VALUES

        return _SPARSE_DICT


    SPARSE_DICT = {}

    for block in ARRAY.blocks:

        # RECHUNK THE DASK ARRAY SO THAT THE CHUNKS CONTAIN FULL INNER VECTORS
        # ARRAY = ARRAY.rechunk((ARRAY.chunks[0], (ARRAY.shape[1],)))
        # 24_04_26_17_34_00 THIS DOES NOT NEED TO BE DONE AS .blocks
        # CONTAINS THE ENTIRE VECTORS FOR EACH ROW-WISE BLOCK (THE
        # COLUMN-WISE CHUNK SIZE APPEARS TO BE IGNORED)

        SUB_DICT = core_inner_dict_builder(block.compute())
        SPARSE_DICT = core_merge_outer(SPARSE_DICT, SUB_DICT)[0]


    if is_inner:
        SPARSE_DICT = SPARSE_DICT[0]


    return SPARSE_DICT



def zip_datadict(DATADICT:dict, dtype=float):
    """Convert datadict as {str:list, str:list, ...} to a sparse dictionary.

    Parameters
    ----------
    DATADICT:
        dict(str:list) - The datadict to convert to a sparse dictionary.
    dtype:
        [int, float, np.int8, np.int16, np.int32, np.int64, np.uint8,
        np.uint16, np.uint32, np.uint64, np.float16, np.float32, np.float64],
        default = float - the python or numpy dtype of the values in the
        returned sparse dictionary. See Notes. Dictionary keys are always
        python integers and cannot be changed.

    Return
    ------
    SPARSE_DICT, HEADER:
        tuple(SPARSE_DICT:dict, header:ndarray) - The sparse dictionary
            representation of the original datadict object and the header
            extracted from it as ndarray.

    Notes
    -----
    There is substantial performance drop-off when using non-python dtypes as
    values in dictionaries. Although use of numpy dtypes is available, the
    recommended dtypes are standard python 'int' and 'float'.

    Examples
    --------
    >>> from pybear.sparse_dict import zip_datadict
    >>> raw_data = np.array([[0,1],[1,1],[0,0],[1,0]]).transpose()
    >>> columns = list('AB')
    >>> data = dict((zip(columns, raw_data)))
    >>> SD, HEADER = zip_datadict(data, dtype=int)
    >>> print(SD)
    {0: {1: 1}, 1: {0: 1, 1: 1}, 2: {1: 0}, 3: {0: 1, 1: 0}}
    >>> print(HEADER)
    [['A', 'B']]

    """

    val._insufficient_datadict_args_1(DATADICT)
    DATADICT, HEADER = val._datadict_init(DATADICT)
    # _datadict_init EXTRACTS KEYS AS HEADER AND CONVERTS DATADICT KEYS TO INT

    SPARSE_DICT = {}

    if len(DATADICT) == 0:
        return SPARSE_DICT, HEADER
    elif len(DATADICT) == 1 and len(DATADICT[0]) == 0:
        return {0: {}}, HEADER
    else:
        for key in range(len(DATADICT)):
            _ = DATADICT[key]
            #   ik = inner_key, v = values, ilt = inner_list_type
            SPARSE_DICT[int(key)] = {int(ik): dtype(v) for
                           ik, v in enumerate(_) if (v != 0 or ik == len(_) - 1)}

    # COMES IN AS {'header': {COLUMNAR LIST}}, TRANSPOSE TO {} = ROWS
    SPARSE_DICT = core_sparse_transpose(SPARSE_DICT)

    return SPARSE_DICT, HEADER


def zip_dataframe(DATAFRAME: pd.DataFrame, dtype=float):
    """Convert a pandas dataframe to a sparse dictionary.  Returns the sparse
    dictionary and the extracted header as a tuple.

    Parameters
    ----------
    DATAFRAME:
        pandas DataFrame - the object to convert into a sparse dictionary
    dtype:
        [int, float, np.int8, np.int16, np.int32, np.int64, np.uint8,
        np.uint16, np.uint32, np.uint64, np.float16, np.float32, np.float64],
        default = float - the python or numpy dtype of the values in the
        returned sparse dictionary. See Notes. Dictionary keys are always
        python integers and cannot be changed.

    Return
    ------
    SPARSE_DICT, HEADER:
        tuple(SPARSE_DICT:dict, header:ndarray) - The sparse dictionary
        representation of the original dataframe object and the header
        extracted from the dataframe as ndarray.

    See Also
    --------
    pandas.DataFrame

    Notes
    -----
    There is substantial performance drop-off when using non-python dtypes as
    values in dictionaries. Although use of numpy dtypes is available, the
    recommended dtypes are standard python 'int' and 'float'.

    Examples
    --------
    >>> import pandas as pd
    >>> array = np.array([[1,1,1],[0,0,1],[1,0,1],[1,0,0]])
    >>> data = pd.DataFrame(data=array, columns=list('ABC'))
    >>> SD, HEADER = zip_dataframe(data, dtype=int)
    >>> print(SD)
    {0: {0: 1, 1: 1, 2: 1}, 1: {2: 1}, 2: {0: 1, 2: 1}, 3: {0: 1, 2: 0}}
    >>> print(HEADER)
    [['A', 'B', 'C']]

    """

    val._insufficient_dataframe_args_1(DATAFRAME)

    if "DASK" in str(type(DATAFRAME)).upper():
        raise TypeError(f"'DATAFRAME' cannot be a dask object. "
                        f"Use zip_dask_dataframe.")

    DATAFRAME, HEADER = val._dataframe_init(DATAFRAME)

    SPARSE_DICT = zip_array(DATAFRAME.to_numpy(), dtype=dtype)

    return SPARSE_DICT, HEADER


def zip_dask_dataframe(DATAFRAME:ddf.DataFrame, dtype=float):
    """Convert a dask dataframe to a sparse dictionary.  Returns the sparse
    dictionary and the extracted header as a tuple.

    Parameters
    ----------
    DATAFRAME: dask DataFrame - the object to convert into a sparse dictionary
    dtype: [int, float, np.int8, np.int16, np.int32, np.int64, np.uint8,
        np.uint16, np.uint32, np.uint64, np.float16, np.float32, np.float64],
        default = float - the python or numpy dtype of the values in the
        returned sparse dictionary. See Notes. Dictionary keys are always
        python integers and cannot be changed.

    Return
    ------
    SPARSE_DICT, HEADER -
        tuple(SPARSE_DICT:dict, HEADER:ndarray) - The sparse dictionary
        representation of the original dataframe object and the header
        extracted from the dataframe as ndarray.

    See Also
    --------
    pybear.sparse_dict.zip_dataframe
    dask.DataFrame

    Notes
    -----
    There is substantial performance drop-off when using non-python dtypes as
    values in dictionaries. Although use of numpy dtypes is available, the
    recommended dtypes are standard python 'int' and 'float'.

    Examples
    --------
    >>> from pybear.sparse_dict import zip_dask_dataframe
    >>> import dask.dataframe as ddf
    >>> import dask.array as da
    >>> array = da.array([[0,1,1,0],[1,0,0,1],[0,0,0,0]])
    >>> data = ddf.from_array(array)
    >>> data.columns = list('ABCD')
    >>> SD, HEADER = zip_dask_dataframe(data, dtype=int)
    >>> print(SD)
    {0: {1: 1, 2: 1, 3: 0}, 1: {0: 1, 3: 1}, 2: {3: 0}}
    >>> print(HEADER)
    [['A' 'B' 'C' 'D']]

    """

    if not isinstance(DATAFRAME, ddf.DataFrame):
        raise TypeError(f'only accepts dask dataframes')

    HEADER = np.array(DATAFRAME.columns).reshape((1,-1)).tolist()

    # len(EMPTY DDF) WORKS FOR dask 2024.1.1 and 2024.2.1, BUT NOT 2024.3+
    # 24_04_30_10_45_00 pyproject.toml UPDATED TO REFLECT dask = "<2024.3.0"
    # (GIVES TypeError: unsupported operand type(s) for +: 'NoneType' and 'int')
    if len(DATAFRAME) == 0:
        SPARSE_DICT = {}

    else:
        SPARSE_DICT = zip_dask_array(DATAFRAME.to_dask_array(), dtype=dtype)

    return SPARSE_DICT, HEADER



# END ZIP
# ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **
# ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **
# START UNZIP



def unzip_to_ndarray(DICT:dict, dtype=np.float64) -> np.ndarray:

    """Convert sparse dict to ndarray.

    Parameters
    ----------
    DICT:
        dict - sparse dictionary to convert to a numpy array
    dtype:
        [int, float, np.int8, np.int16, np.int32, np.int64, np.uint8,
        np.uint16, np.uint32, np.uint64, np.float16, np.float32, np.float64],
        default = np.float64 - the numpy dtype of the values in the returned
        numpy array.

    Return
    ------
    NDARRAY:
        np.ndarray[, np.ndarray] - The numpy array representation of
        the given sparse dictionary.

    Examples
    -------
    >>> import numpy as np
    >>> from pybear.sparse_dict import unzip_to_ndarray
    >>> sd = {0:{0:1,2:1}, 1:{1:1,2:0}, 2:{0:1,2:1}}
    >>> NDARRAY = unzip_to_ndarray(sd, dtype=np.uint8)
    >>> print(NDARRAY)
    [[1 0 1]
     [0 1 0]
     [1 0 1]]

    """

    val._insufficient_dict_args_1(DICT)

    try:
        np.array([]).astype(dtype)
    except:
        raise TypeError(f"'dtype' must be a valid python or numpy dtype")

    if DICT == {}:
        return np.array([], dtype=dtype)
    elif len(DICT) == 1 and DICT[list(DICT.keys())[0]] == {}:
        return np.array([[]], dtype=dtype)

    DICT = val._dict_init(DICT)

    _dtype = dtype

    _shape = shape_(DICT)

    _is_inner = False
    if len(_shape) == 1:
        DICT = {0: DICT}
        _shape = (1, _shape[0])
        _is_inner = True

    NDARRAY = np.zeros(_shape, dtype=_dtype)
    for outer_key in range(_shape[0]):
        NDARRAY[outer_key][np.fromiter((DICT[outer_key]), dtype=np.int32)] = \
            np.fromiter((DICT[outer_key].values()), dtype=_dtype)

    if _is_inner:
        NDARRAY = NDARRAY[0]

    return NDARRAY


def unzip_to_list(DICT:dict, dtype=float) -> list:

    """Convert sparse dict to list of lists.

    Parameters
    ----------
    DICT:
        dict - sparse dictionary to convert to a list or list-of-lists
    dtype:
        [int, float, np.int8, np.int16, np.int32, np.int64, np.uint8,
        np.uint16, np.uint32, np.uint64, np.float16, np.float32, np.float64],
        default = float - the dtypes of the values in the returned list(s).

    Return
    -------
    LIST_OF_LISTS:
        list[, np.ndarray] - The python list representation of the given
        sparse dictionary.

    See Also
    --------
    pybear.sparse_dict.unzip_to_ndarray

    Examples
    --------
    >>> import numpy as np
    >>> from pybear.sparse_dict import unzip_to_list
    >>> sd = {0:{0:1,2:1}, 1:{1:1,2:0}, 2:{0:1,2:1}}
    >>> LIST_OF_LISTS = unzip_to_list(sd, dtype=np.uint8)
    >>> print(LIST_OF_LISTS)
    [[1, 0, 1], [0, 1, 0], [1, 0, 1]]

    """

    try:
        np.array([]).astype(dtype)
    except:
        raise TypeError(f"'dtype' must be a valid python or numpy dtype")


    _dtype = dtype

    NDARRAY = unzip_to_ndarray(DICT)

    if len(NDARRAY.shape)==1:
        LIST_OF_LISTS = list(map(_dtype, NDARRAY))
        del NDARRAY
    elif len(NDARRAY.shape)==2:
        LIST_OF_LISTS = list(NDARRAY)
        del NDARRAY
        for idx in range(len(LIST_OF_LISTS)):
            LIST_OF_LISTS[idx] = list(map(_dtype, LIST_OF_LISTS[idx]))
    else:
        raise ValueError(f"ndarray with unhandled shape {NDARRAY.shape}")

    return LIST_OF_LISTS


def unzip_to_dask_array(
    DICT:dict,
    dtype=np.float64,
    chunks=None
) -> da.array:

    """Convert sparse dictionary to a dask array.

    Parameters
    ----------
    DICT:
        dict - sparse dictionary to convert to a dask array
    dtype:
        [int, float, np.int8, np.int16, np.int32, np.int64, np.uint8,
        np.uint16, np.uint32, np.uint64, np.float16, np.float32, np.float64],
        default = np.float64 - the dtypes of the values in the returned
        dask array.
    chunks:
        tuple, default = None - Chunk shape; the resulting dask array
        representation is rechunked by this tuple. If no value is passed, the
        dask array is set to one chunk, i.e., chunk shape = DASK_ARRAY.shape

    Return
    -------
    DASK_ARRAY:
        dask.array.array[, np.ndarray] - The dask array representation of the
        given sparse dictionary.

    See Also
    --------
    pybear.sparse_dict.unzip_to_ndarray

    Examples
    --------
    >>> import numpy as np
    >>> from pybear.sparse_dict import unzip_to_dask_array
    >>> sd = {0:{0:1,2:1}, 1:{1:1,2:0}, 2:{0:1,2:1}}
    >>> DASK_ARRAY = unzip_to_dask_array(sd, dtype=np.uint8)
    >>> DASK_ARRAY
    dask.array<setitem, shape=(3, 3), dtype=uint8, chunksize=(3, 3), chunktype=numpy.ndarray>

    >>> DASK_ARRAY.compute()
    array([[1, 0, 1],
           [0, 1, 0],
           [1, 0, 1]], dtype=uint8)

    """

    try:
        np.array([]).astype(dtype)
    except:
        raise TypeError(f"'dtype' must be a valid python or numpy dtype")

    val._insufficient_dict_args_1(DICT)

    if DICT == {}:
        return da.array([], dtype=dtype)
    elif len(DICT) == 1 and DICT[list(DICT.keys())[0]] == {}:
        return da.array([[]], dtype=dtype)

    DICT = val._dict_init(DICT)

    _dtype = dtype

    _shape = shape_(DICT)

    if chunks is None:
        _chunks = _shape
    else:
        if not isinstance(chunks, tuple):
            raise TypeError(f"'chunks' must be a tuple with dimensionality less "
                            f"than or equal to the dimensionality of 'DICT'")

        if len(chunks)==0:
            raise ValueError(f"'chunks' cannot be empty")
        elif len(chunks) > 2:
            raise ValueError(f"chunks dimensionality must be less than or equal "
                             f"to the dimensionality of 'DICT'")

        if 0 in chunks:
            raise ValueError(f"chunk dimension cannot be zero")

        # IF GET TO THIS POINT, GOOD TO GO
        _chunks = chunks

    _is_inner = False
    if len(_shape) == 1:
        DICT = {0: DICT}
        _shape = (1, _shape[0])
        _chunks = (1, _chunks[0])
        _is_inner = True

    DASK_ARRAY = da.zeros(_shape, dtype=_dtype, chunks=_chunks)
    for outer_key in range(_shape[0]):
        # 24_04_27_13_52_00 --- DASK ARRAY ISNT TAKING ASSIGNMENT LIKE
        # DA[0][3] OR DA[0, 3], SO DO THIS OUT THE LONG WAY
        NEW_INNER = DASK_ARRAY[outer_key]
        NEW_INNER[np.fromiter((DICT[outer_key]), dtype=np.int32)] = \
                        np.fromiter((DICT[outer_key].values()), dtype=_dtype)
        DASK_ARRAY[outer_key] = NEW_INNER
        del NEW_INNER


    if _is_inner:
        DASK_ARRAY = DASK_ARRAY[0]

    return DASK_ARRAY


def unzip_to_datadict(
                      DICT:dict,
                      HEADER:[list, np.ndarray, set, tuple]=None,
                      dtype=float
    ) -> dict:

    """Convert sparse dictionary to a datadict of numpy ndarrays.

    Parameters
    ----------
    DICT:
        dict - sparse dictionary to convert to a data dictionary
            dict(str:np.ndarray, str:np.ndarray, ... )
    HEADER:
        [None, list, tuple, set, np.ndarray], default=None - Optional header
        of the sparse dictionary. Orientation of the sparse dictionary is not
        assumed, so the dimension of the header must match either of the axes
        of the sparse dictionary, and cannot be empty. If a header is not
        passed, the columns of the datadict are keyed numerically in this way:
        '0', '1', '2', etc.
    dtype:
        [int, float, np.int8, np.int16, np.int32, np.int64, np.uint8,
        np.uint16, np.uint32, np.uint64, np.float16, np.float32, np.float64],
        default = float - the dtypes of the values in the returned array(s).

    Returns
    -------
    DATADICT:
        dict - The data dictionary representation of the given sparse
        dictionary and its header. A header is not returned separately.

    Examples
    --------
    >>> import numpy as np
    >>> from pybear.sparse_dict import unzip_to_datadict
    >>> sd = {0:{0:1,2:1}, 1:{1:1,2:0}, 2:{0:1,2:1}}
    >>> columns = list('ABC')
    >>> DATADICT = unzip_to_datadict(sd, HEADER=columns, dtype=np.uint8)
    >>> print(DATADICT)
    {'A': array([1, 0, 1], dtype=uint8), 'B': array([0, 1, 0], dtype=uint8), 'C': array([1, 0, 1], dtype=uint8)}


    """


    def _validate_header(_HEADER, __shape):

        if _HEADER is not None:
            try:
                iter(_HEADER)
                if isinstance(_HEADER, (dict, str)):
                    raise UnicodeError  # SOME OBSCURE ERROR
            except (TypeError, UnicodeError):
                raise TypeError(f"'HEADER' must be an array-like")
            except Exception as e:
                raise AssertionError(f"iter check failed for reason other than "
                                    f"TypeError --- {e}")

            _HEADER = np.array(list(_HEADER)).ravel()

            if len(_HEADER) not in __shape:
                raise ValueError(f'number of header positions ({len(_HEADER)}) '
                 f'must match one of the dimensions of the sparse dict {__shape}')

        elif _HEADER is None:  # ASSUME DICT WAS PASSED INTENDING ORIENTED AS ROWS
            _HEADER = np.fromiter(map(str, list(range(inner_len(DICT)))), dtype='<U5')

        return _HEADER


    val._insufficient_dict_args_1(DICT)


    if DICT == {}:
        return {}

    elif len(DICT) == 1 and DICT[list(DICT.keys())[0]] == {}:
        HEADER = _validate_header(HEADER, (1,0))
        return {HEADER.ravel()[0]: []}


    DICT = val._dict_init(DICT)

    if val._is_sparse_inner(DICT):
        DICT = {0: DICT}
        DICT = core_sparse_transpose(DICT)  # IN THE ABSENCE OF A HEADER, AN
        # INNER DICT SHOULD BE HANDLED AS IF IT IS A COLUMN OF DATA (WHEREAS IF
        # IT WERE HANDLED LIKE A FULL DICT IT WOULD BE TREATED AS A ROW)

    _dtype = dtype

    _shape = shape_(DICT)

    HEADER = _validate_header(HEADER, _shape)

    del _validate_header

    NDARRAY = unzip_to_ndarray(DICT).astype(_dtype)

    # FIND WHICH AXIS THE HEADER ALIGNS WITH AND ALIGN NDARRAY SO THAT DATA IS
    # COLUMNAR
    if len(HEADER) == _shape[0]:
        pass
    elif len(HEADER) == _shape[1]:
        NDARRAY = NDARRAY.transpose()

    DATADICT = dict((zip(HEADER, NDARRAY)))

    return DATADICT


def unzip_to_dense_dict(
                        DICT:dict,
                        dtype=float
    ) -> dict:

    """Convert sparse dictionary to a dense dictionary. All array positions are
    represented in the dictionary, including zeros.

    Parameters
    ----------
    DICT:
        dict - the sparse dictionary to convert to a dense dictionary.
    dtype:
        [int, float, np.int8, np.int16, np.int32, np.int64, np.uint8,
        np.uint16, np.uint32, np.uint64, np.float16, np.float32, np.float64],
        default = float - the dtypes of the values in the returned dense
        dictionary.

    Return
    ------
    DENSE_DICT:
        dict - dense dictionary representation of the given sparse dictionary.

    Examples
    --------
    >>> from pybear.sparse_dict import unzip_to_dense_dict
    >>> data = {0: {1: 1, 2: 0}, 1: {2: 3}}
    >>> out = unzip_to_dense_dict(data, dtype=int)
    >>> print(out)
    {0: {0: 0, 1: 1, 2: 0}, 1: {0: 0, 1: 0, 2: 3}}

    """

    val._insufficient_dict_args_1(DICT)

    if DICT == {}:
        return {}
    elif len(DICT) == 1 and DICT[list(DICT.keys())[0]] == {}:
        return DICT

    DICT = val._dict_init(DICT)

    DICT = clean(DICT)

    _is_inner = False
    if len(shape_(DICT)) == 1:
        _is_inner = True
        DICT = {0: DICT}

    _dtype = dtype

    _outer_len, _inner_len = shape_(DICT)

    NDARRAY = unzip_to_ndarray(DICT, _dtype)

    DENSE_DICT = {}

    for outer_key in range(_outer_len):

        if _dtype in (int, float):
            VALUES = map(_dtype, NDARRAY[outer_key].tolist())
        else:
            VALUES = NDARRAY[outer_key]

        DENSE_DICT[int(outer_key)] = dict((zip(range(_inner_len), VALUES)))

    if _is_inner:
        DENSE_DICT = DENSE_DICT[0]

    return DENSE_DICT


def unzip_to_dataframe(
                        DICT:dict,
                        HEADER:[list, np.ndarray, set, tuple, None]=None,
                        dtype=np.float64
    ) -> pd.DataFrame:

    """Convert sparse dictionary to a pandas dataframe.

    Parameters
    ----------
    DICT:
        dict - the sparse dictionary to convert into a pandas dataframe
    HEADER:
        [None, list, tuple, set, np.ndarray], default=None - Optional header
        of the sparse dictionary. Orientation of the sparse dictionary is not
        assumed, so the dimension of the header must match either of the axes
        of the sparse dictionary, and cannot be empty. If a header is not
        passed, the columns of the dataframe are keyed numerically in this way:
        '0', '1', '2', etc.
    dtype:
        [int, float, np.int8, np.int16, np.int32, np.int64, np.uint8,
        np.uint16, np.uint32, np.uint64, np.float16, np.float32, np.float64],
        default = float - the dtypes of the values in the returned dataframe.

    Return
    ------
    DATAFRAME:
        pandas.DataFrame representation of the original sparse dictionary


    Examples
    --------
    >>> from pybear.sparse_dict import unzip_to_dataframe
    >>> data = {0:{1: 2}, 1:{0:1,1:1}, 2:{0:2,1:0}, 3:{0:2,1:0}}
    >>> columns = ['A', 'B']
    >>> out = unzip_to_dataframe(data, HEADER=columns, dtype=np.uint8)
    >>> print(out)
       A  B
    0  0  2
    1  1  1
    2  2  0
    3  2  0


    """

    _dtype = dtype

    DATADICT = unzip_to_datadict(DICT, HEADER=HEADER, dtype=_dtype)

    DATAFRAME = pd.DataFrame(DATADICT, dtype=_dtype)

    return DATAFRAME



def unzip_to_dask_dataframe(
                            DICT:dict,
                            chunksize:int=50_000,
                            HEADER:[list, np.ndarray, set, tuple, None]=None,
                            dtype=np.float64
    ) -> ddf.DataFrame:

    """Convert sparse dictionary to a dask dataframe.

    Parameters
    ----------
    DICT:
        dict - the sparse dictionary to convert into a dask dataframe
    HEADER:
        [None, list, tuple, set, np.ndarray], default=None - Optional header
        of the sparse dictionary. Orientation of the sparse dictionary is not
        assumed, so the dimension of the header must match either of the axes
        of the sparse dictionary, and cannot be empty. If a header is not
        passed, the columns of the dataframe are keyed numerically in this way:
        '0', '1', '2', etc.
    chunksize:
        [int, tuple], default = 50_000 - partition size; the resulting dask
        dataframe is partitioned by this number. If passed as a tuple, only
        the first value is taken as the partition size.
    dtype:
        [int, float, np.int8, np.int16, np.int32, np.int64, np.uint8,
        np.uint16, np.uint32, np.uint64, np.float16, np.float32, np.float64],
        default = float - the dtypes of the values in the returned dataframe.

    Return
    ------
    DASK_DATAFRAME:
        dask.dataframe.DataFrame representation of the original sparse dictionary

    See Also
    --------
    pybear.sparse_dict.unzip_to_dataframe
    dask.dataframe.DataFrame

    Examples
    --------
    >>> from pybear.sparse_dict import unzip_to_dask_dataframe
    >>> data = {0:{1: 2}, 1:{0:1,1:1}, 2:{0:2,1:0}, 3:{0:2,1:0}}
    >>> columns = ['A', 'B']
    >>> out = unzip_to_dask_dataframe(data, HEADER=columns, dtype=np.uint8)
    >>> type(out)
    <class 'dask.dataframe.core.DataFrame'>

    >>> print(out.compute())
       A  B
    0  0  2
    1  1  1
    2  2  0
    3  2  0

    """

    def _validate_header(_HEADER, __shape):

        if _HEADER is not None:
            try:
                iter(_HEADER)
                if isinstance(_HEADER, (dict, str)):
                    raise UnicodeError  # SOME DUMMY ERROR
            except (TypeError, UnicodeError):
                raise TypeError(f"'HEADER' must be an array-like")
            except Exception as e:
                raise AssertionError(f"iter check failed for reason other than "
                     f"TypeError --- {e}")

            _HEADER = np.array(list(HEADER)).ravel()

            if len(_HEADER) not in __shape:
                raise ValueError(f'number of header positions ({len(_HEADER)}) '
                f'must match one of the dimensions of the sparse dict {__shape}')

        elif _HEADER is None:
            # ASSUME DICT WAS PASSED INTENDING ORIENTED AS ROWS
            _HEADER = np.fromiter(map(str, list(range(inner_len(DICT)))),
                                  dtype='<U5')

        return _HEADER


    val._insufficient_dict_args_1(DICT)

    # validate chunksize ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
    err_msg = (f"'chunksize' must be an integer or a 1-D list-like of at least "
               f"one integer, and must be > 0")
    try:
        iter(chunksize)
        if isinstance(chunksize, (str, dict, set)):
            raise UnicodeError  # SOME OBSCURE ERROR

        try:
            list(map(float, chunksize))
        except:
            raise UnicodeError  # SOME OBSCURE ERROR

        chunksize = chunksize[0]

    except UnicodeError:
        raise TypeError(err_msg) from None
    except TypeError:
        try:
            float(chunksize)
            if isinstance(chunksize, bool):
                raise ValueError
        except ValueError:
            raise TypeError(err_msg) from None
    except IndexError:
        raise ValueError(err_msg)
    except Exception as e:
        raise Exception(f"chunksize iter validation excepted for uncontrolled "
                        f"reason --- {e}")

    try:
        val._is_int(chunksize)
        if chunksize < 1:
            raise TypeError
    except TypeError:
        raise ValueError(err_msg) from None
    except Exception as e:
        raise Exception(f"chunksize integer validation raised error other than "
                            f"Type error --- {e}")
    del err_msg

    _chunksize = chunksize
    # end validate chunksize ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    if DICT == {}:
        return ddf.from_pandas(
                               pd.DataFrame({}, dtype=dtype),
                               chunksize=_chunksize
        )

    elif len(DICT) == 1 and DICT[list(DICT.keys())[0]] == {}:
        HEADER = _validate_header(HEADER, (1,0))
        return ddf.from_pandas(
                                pd.DataFrame({}, columns=HEADER, dtype=dtype),
                                chunksize=_chunksize
        )


    DICT = val._dict_init(DICT)

    if len(shape_(DICT)) == 1:
        DICT = {0: DICT}
        DICT = core_sparse_transpose(DICT)  # IN THE ABSENCE OF A HEADER, AN
        # INNER DICT SHOULD BE HANDLED AS IF IT IS A COLUMN OF DATA (WHEREAS IF
        # IT WERE HANDLED LIKE A FULL DICT IT WOULD BE TREATED AS A ROW)

    _dtype = dtype

    _shape = shape_(DICT)

    HEADER = _validate_header(HEADER, _shape)

    del _validate_header

    # FIND WHICH AXIS THE HEADER ALIGNS WITH AND ALIGN DICT SO THAT DATA IS
    # ORIENTED IN ROWS
    if len(HEADER) == _shape[1]:
        pass
    elif len(HEADER) == _shape[0]:
        DICT = core_sparse_transpose(DICT)
        _shape = (_shape[1], _shape[0])

    DASK_ARRAY = unzip_to_dask_array(DICT, dtype=_dtype, chunks=(chunksize,_shape[1]))

    DASK_DATAFRAME = ddf.from_dask_array(DASK_ARRAY, columns=HEADER)

    del DASK_ARRAY

    return DASK_DATAFRAME













