# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import numpy as np
from pybear.sparse_dict._validation import (
                                            _insufficient_dict_args_1 as ida1,
                                            _insufficient_dict_args_2 as ida2,
                                            _is_sparse_inner,
                                            _is_sparse_outer,
                                            _dict_init
)



def get_shape(
                name:str,
                OBJECT:[np.ndarray, dict],
                given_orientation:str
    ) -> tuple:

    """Return shape in numpy format for array-like and sparse dicts, except
    (x,y) x is always rows and y is always columns w-r-t to given orientation.

    Parameters
    ----------
    name:
        str - name of the object
    OBJECT:
        [list, tuple, np.ndarray, dict] - the object to measure shape of,
        cannot be greater tnan 2D.
    given_orientation:
        ['row', 'column'] - the orientation of the object; the conventional
        python orientation is 'row', where the comprising vectors represent
        rows of data.

    Return
    ------
    _shape:
        tuple - the lengths of the object in the (row, column) dimensions.

    See Also
    --------
    pybear.sparse_dict.shape_
    numpy.ndarray.shape

    Examples
    --------
    >>> from pybear.sparse_dict import get_shape
    >>> SD = {0:{0:1, 3:0}, 1:{3:1}}
    >>> out = get_shape('sd', SD, given_orientation='row')
    >>> out
    (2, 4)

    >>> import numpy as np
    >>> NP = np.random.randint(0, 10, (3,4))
    >>> out = get_shape('np', NP, given_orientation='column')
    >>> out
    (4, 3)

    """

    if name is None or not isinstance(name, str):
        raise TypeError(f"'name' MUST BE PASSED AND MUST BE A str")

    if OBJECT is None:
        return ()    # THIS IS THE OUTPUT THAT NUMPY GIVES FOR None

    err_msg = (f"'OBJECT' must be an array-like or sparse dict with at most 2 "
               f"dimensions")

    try:
        iter(OBJECT)
    except:
        raise TypeError(err_msg)

    if isinstance(OBJECT, str):
        raise TypeError(err_msg)
    elif isinstance(OBJECT, set):
        raise TypeError(err_msg + f'. Cannot be a set.')
    elif isinstance(OBJECT, dict):
        _shape = shape_(OBJECT)
    else:
        try:
            _shape = np.array(OBJECT).shape
        except:
            raise TypeError(f"if 'OBJECT' is not a sparse dictionary it must "
                            f"be able to convert to a numpy array")


    # IF OBJECT IS PASSED AS SINGLE [] OR {}, DONT WORRY ABOUT ORIENTATION
    if len(_shape)==1:
        return _shape

    # CAN ONLY GET HERE IF len(_shape) >= 2
    err_msg = f'given_orientation MUST BE PASSED, AND MUST BE "ROW" OR "COLUMN"'
    try:
        given_orientation = given_orientation.upper()
    except:
        raise TypeError(err_msg)

    if given_orientation not in ['ROW', 'COLUMN']:
        raise ValueError(f'INVALID given_orientation "{given_orientation}"')


    #if given_orientation == 'ROW': _shape STAYS AS _shape
    if given_orientation == 'COLUMN': # _shape GETS REVERSED
        _shape = _shape[1], _shape[0]

    return _shape


def outer_len(DICT1:dict) -> int:

    """Length of the outer dictionary that holds the inner dictionaries as
    values, i.e., number of inner dictionaries. The dictionary keys along this
    dimension must be zero-based and fully dense for correctly built sparse
    dictionaries, thus the length of this dimension of the dictionary is the
    true magnitude of this axis of the array.

    Parameters
    ---------
    DICT1:
        dict - object to get outer length of

    Return
    ------
    outer_len:
        int - outer length, i.e., number of inner dictionaries

    Examples:
    >>> from pybear.sparse_dict import outer_len
    >>> SD = {0: {2:1}, 1:{2:2}, 2:{2:3}}
    >>> out = outer_len(SD)
    >>> out
    3

    """

    # DONT BOTHER TO clean() OR resize() HERE, SINCE ONLY THE SCALAR LENGTH IS
    # RETURNED (CHANGES TO DICT1 ARENT RETAINED)
    ida1(DICT1)
    DICT1 = _dict_init(DICT1)

    if DICT1 == {}: return (0,)

    if _is_sparse_inner(DICT1):
        raise ValueError(f'cannot get outer len for an inner dict')

    try:
        outer_len = len(DICT1)
    except:
        # ???
        raise ValueError(f'Sparse dictionary is a zero-len outer dictionary')

    return outer_len


def inner_len_quick(DICT1:dict) -> int:

    """Spatial length (not representational length) of inner dictionaries that
    are held by the outer dictionary. Assumes a clean sparse dict and bypasses
    most validation for speed - use with caution.

    Parameters
    ----------
    DICT1:
        dict - object for which to get inner length

    Return
    ------
    inner_len:
        int - magnitude of the inner dimension.

    See Also
    --------
    pybear.sparse_dict.inner_len

    Examples
    --------
    >>> from pybear.sparse_dict import inner_len_quick
    >>> SD = {0: {0:1, 2:2}, 1: {1:3, 2:0}}
    >>> out = inner_len_quick(SD)
    >>> out
    3

    """

    if not isinstance(DICT1, dict):
        raise TypeError(f'DICT1 must be a sparse dictionary')

    try:
        inner_len = max(DICT1[0]) + 1
    except:
        try:
            inner_len = max(DICT1) + 1
        except:
            raise ValueError(f'unable to get inner len')

    return inner_len


def inner_len(DICT1):

    """Spatial Length (not representational length) of inner dictionaries that
    are held by the outer dictionary, as determined by the value of the largest
    key in each inner dictionary. For a properly constructed sparse dictionary,
    the largest key in every inner dictionary must be the same, with no greater
    keys. This axis may be sparse and the physical length of the inner
    dictionary may not reflect the true scale of the axis in space, therefore
    the largest key is used.

    Parameters
    ----------
    DICT1:
        dict - object for which to get inner length

    Return
    ------
    inner_len:
        int - magnitude of the inner dimension

    See Also
    --------
    pybear.sparse_dict.inner_len_quick

    Examples
    --------
    >>> from pybear.sparse_dict import inner_len
    >>> SD = {0: {3:1, 5:0}, 1: {1:2, 5:1}}
    >>> out = inner_len(SD)
    >>> out
    6

    """

    # DONT BOTHER TO clean() OR resize() HERE, SINCE ONLY THE SCALAR LENGTH IS
    # RETURNED (CHANGES TO DICT1 ARENT RETAINED)
    ida1(DICT1)
    DICT1 = _dict_init(DICT1)

    if _is_sparse_outer(DICT1):
        # IF ALL INNERS ARE EMPTY
        if all([True if DICT1[outer_idx] == {} else False for outer_idx in DICT1]):
            inner_len = 0
        else:
            try:
                inner_len = max(map(max, DICT1.values())) + 1
            except:
                # IF AN INNER IS EMPTY BUT OTHERS ARE NOT (SHOULD HAVE PLACEHOLDER)
                raise Exception(f'Sparse dictionary has a zero-len inner dictionary')
    elif _is_sparse_inner(DICT1):
        if len(DICT1) == 0:
            inner_len = 0
        else:
            inner_len = max(DICT1) + 1
    else:
        raise ValueError(f'Invalid sparse dictionary object')

    return inner_len


def size_(DICT1:dict) -> int:

    """Return outer dict length * inner dict length. The number of elements if
    converted to a dense representation.

    Parameters
    ----------
    DICT1:
        dict - the object for which to measure size


    Return
    ------
    size_:
        int - the total number of sparse and dense elements; the total number
        of elements.

    See Also
    --------
    numpy.size

    Examples
    --------
    >>> from pybear.sparse_dict import size_
    >>> SD = {0: {0:1, 1:8, 2:0}, 1: {2: 4}}
    >>> out = size_(SD)
    >>> out
    6


    """

    if _is_sparse_outer(DICT1):
        size_ =  outer_len(DICT1) * inner_len(DICT1)
    elif _is_sparse_inner(DICT1):
        size_ = inner_len(DICT1)
    elif isinstance(DICT1, dict):
        raise ValueError(f"invalid sparse dictionary format")
    else:
        raise TypeError(f"Invalid type '{type(DICT1)}'")

    return size_


def shape_(DICT1:dict) -> tuple:
    """Returns shape of sparse dictionary using numpy shape rules ---
    (outer dict length, inner dict length) --- as tuple.

    Parameters
    ----------
    DICT1:
        dict - Object for which to measure shape

    Return
    ------
    shape:
        tuple - (outer length, inner length)

    See Also
    --------
    numpy.shape

    Examples
    --------
    >>> from pybear.sparse_dict import shape_
    >>> SD = {0: {99: 0}, 1: {99: 0}, 2: {99: 0}}
    >>> out = shape_(SD)
    >>> out
    (3, 100)

    """

    if DICT1 is None:
        return ()
    if DICT1 == {}:
        return (0,)

    ida1(DICT1)

    if _is_sparse_inner(DICT1):
        # ASSUMES PLACEHOLDER RULE IS USED (len = max key + 1)
        shape = (max(DICT1.keys())+1,)

    elif _is_sparse_outer(DICT1):
        # AFTER HERE ALL MUST BE DICT OF INNER DICTS
        # IF ALL INNERS ARE {}  (ZERO len) ** * ** * ** * ** * ** * ** * ** *
        _LENS = list(map(len, DICT1.values()))
        if (min(_LENS)==0) and (max(_LENS)==0):
            shape = (len(DICT1), 0)
            return shape
        del _LENS
        # END IF ALL INNERS ARE {}  (ZERO len) ** * ** * ** * ** * ** * ** *

        # TEST FOR RAGGEDNESS ** * ** * ** * ** * ** * ** * ** * ** * ** * **
        _INNER_LENS = list(map(max, DICT1.values()))
        if not min(_INNER_LENS)==max(_INNER_LENS):
            raise ValueError(f'DICT1 is ragged')
        # END TEST FOR RAGGEDNESS ** * ** * ** * ** * ** * ** * ** * ** * ** *

        shape = (len(DICT1), _INNER_LENS[0]+1)

    else:
        raise TypeError(f'invalid sparse dictionary format')

    return shape


def clean(DICT1:dict) -> dict:

    """Remove any non-placeholding zero values, enforce contiguous zero-indexed
    outer keys, insert any missing placeholding zeros.

    Parameters
    ----------
    DICT1:
        dict - object to clean

    Return
    ------
    DICT1:
        dict - cleaned DICT1

    Examples
    --------
    >>> from pybear.sparse_dict import clean
    >>> BAD_SD = {0:{0:0, 1:0, 2:1, 3:0}, 2:{0:0, 4:1}}
    >>> GOOD_SD = clean(BAD_SD)
    >>> GOOD_SD
    {0: {2: 1, 4: 0}, 1: {4: 1}}

    """

    if DICT1 == {}:
        return DICT1

    ida1(DICT1)


    def inner_dict_zero_scrubber(inner_dict):
        VALUES_AS_NP = np.fromiter(inner_dict.values(), dtype=float)
        if 0 in VALUES_AS_NP[:-1]:
            KEYS_AS_NP = np.fromiter(inner_dict, dtype=int)
            KEYS_OF_ZEROES = KEYS_AS_NP[
                np.argwhere(VALUES_AS_NP[:-1] == 0).transpose()[0]
            ]
            np.fromiter((inner_dict.pop(_) for _ in KEYS_OF_ZEROES), dtype=object)
            del KEYS_AS_NP, KEYS_OF_ZEROES

        del VALUES_AS_NP

        return inner_dict


    if _is_sparse_outer(DICT1):
        # CHECK IF KEYS START AT ZERO AND ARE CONTIGUOUS
        # FIND ACTUAL len, SEE IF OUTER KEYS MATCH EVERY POSN IN THAT RANGE
        # IF NOT, REASSIGN keys TO DICT, IDXed TO 0
        if not all((_ in DICT1 for _ in range(len(DICT1)))):
            for _, __ in enumerate(list(DICT1.keys())):
                DICT1[int(_)] = DICT1.pop(__)

        max_inner_key = max(map(max, DICT1.values()))
        for outer_key in DICT1:
            # ENSURE INNER DICT PLACEHOLDER RULE
            # (KEY FOR LAST POSN, EVEN IF VALUE IS ZERO) IS ENFORCED
            DICT1[int(outer_key)][int(max_inner_key)] = \
                DICT1[outer_key].get(max_inner_key, 0)
            # ENSURE THERE ARE NO ZERO VALUES IN ANY INNER DICT EXCEPT LAST POSN
            DICT1[int(outer_key)] = inner_dict_zero_scrubber(DICT1[outer_key])

    elif _is_sparse_inner(DICT1):
        DICT1 = inner_dict_zero_scrubber(DICT1)

    elif isinstance(DICT1, dict):
        raise ValueError(f'invalid sparse dictionary format')

    else:
        raise TypeError(f"invalid object type '{type(DICT1)}'")

    del inner_dict_zero_scrubber

    return DICT1


def sparsity(DICT1:dict) -> float:

    """Calculate the sparsity of a sparse dictionary.

    Parameters
    ----------
    DICT1:
        dict - object for which to measure sparsity

    Return
    ------
    sparsity:
        float - percent of size that is occupied by zeroes in a dense
        representation. Can also be intepreted as the probability of any
        given single element being zero.

    See Also
    --------
    pybear.utils.array_sparsity

    Examples
    --------
    >>> from pybear.sparse_dict import sparsity
    >>> SD = {0: {0:1, 1:0}, 1:{1:1}}
    >>> out = sparsity(SD)
    >>> out
    50.0

    """

    ida1(DICT1)
    DICT1 = _dict_init(DICT1)
    SIZE = size_(DICT1)

    if _is_sparse_outer(DICT1):
        total_hits = 0
        for outer_key in DICT1:
            total_hits += len(DICT1[outer_key]) - int(0 in DICT1[outer_key].values())
    elif _is_sparse_inner(DICT1):
        total_hits = len(DICT1) - int(0 in DICT1.values())

    return 100 - 100 * total_hits / SIZE


def core_sparse_equiv(DICT1:dict, DICT2:dict) -> bool:

    """Check for equivalence of two sparse dictionaries without safeguards for
    speed. Validation to ensure properly constructed sparse dictionaries is not
    performed.

    Parameters
    ---------
    DICT1:
        dict - sparse dictionary to compare against DICT2
    DICT2:
        dict - sparse dictionary to compare against DICT1

    Return
    -----
    bool:
        bool - Equivalence of the given sparse dictionaries.

    See Also
    --------
    pybear.sparse_dict.safe_sparse_equiv

    Examples
    --------
    >>> from pybear.sparse_dict import core_sparse_equiv
    >>> SD1 = {0:{0:0, 1:2, 2:1, 3:0}, 2:{0:0, 4:1}}
    >>> SD2 = {0:{0:2, 1:0, 2:1, 3:1}, 2:{0:1, 4:0}}
    >>> out = core_sparse_equiv(SD1, SD2)
    >>> out
    False

    >>> SD1 = {0:{0:1, 1:2, 2:3}}
    >>> SD2 = {0:{0:1, 1:2, 2:3}}
    >>> out = core_sparse_equiv(SD1, SD2)
    >>> out
    True

    """

    if _is_sparse_inner(DICT1):
        DICT1 = {0: DICT1}

    if _is_sparse_inner(DICT2):
        DICT2 = {0: DICT2}


    # 1) TEST OUTER SIZE
    if len(DICT1) != len(DICT2): return False

    # 2) TEST INNER SIZES
    for outer_key in DICT1:  # ALREADY ESTABLISHED OUTER KEYS ARE EQUAL
        if len(DICT1[outer_key]) != len(DICT2[outer_key]): return False

    # 3) TEST INNER KEYS ARE EQUAL
    for outer_key in DICT1:   # ALREADY ESTABLISHED OUTER KEYS ARE EQUAL
        if not np.array_equiv(DICT1[outer_key], DICT2[outer_key]):
            return False

    # 4) TEST INNER VALUES ARE EQUAL
    for outer_key in DICT1:  # ALREADY ESTABLISHED OUTER KEYS ARE EQUAL
        if not np.allclose(np.fromiter(DICT1[outer_key].values(), dtype=np.float64),
                          np.fromiter(DICT2[outer_key].values(), dtype=np.float64),
                           rtol=1e-8, atol=1e-8
        ):
            return False

    # IF GET THIS FAR, MUST BE True
    return True


def safe_sparse_equiv(DICT1:dict, DICT2:dict) -> bool:

    """Check for equivalence of two sparse dictionaries with safeguards.
    Ensures that both sparse dictionaries are constructed by the rules for
    sparse dictionaries.

    Parameters
    ---------
    DICT1:
        dict - sparse dictionary to compare against DICT2
    DICT2:
        dict - sparse dictionary to compare against DICT1

    Return
    -----
    bool:
        bool - Equivalence of the given sparse dictionaries.

    See Also
    --------
    pybear.sparse_dict.core_sparse_equiv

    Examples
    --------
    >>> from pybear.sparse_dict import core_sparse_equiv
    >>> SD1 = {0:{0:0, 1:2, 2:1, 3:0}, 2:{0:0, 4:1}}
    >>> SD2 = {0:{0:2, 1:0, 2:1, 3:1}, 2:{0:1, 4:0}}
    >>> out = core_sparse_equiv(SD1, SD2)
    >>> out
    False

    >>> SD1 = {0:{0:1, 1:2, 2:3}}
    >>> SD2 = {0:{0:1, 1:2, 2:3}}
    >>> out = core_sparse_equiv(SD1, SD2)
    >>> out
    True

    """


    DICT1 = _dict_init(DICT1)
    DICT2 = _dict_init(DICT2)
    ida2(DICT1, DICT2)

    # 1) TEST OUTER KEYS ARE EQUAL
    if not np.array_equiv(DICT1, DICT2):
        return False

    # 2) RUN core_sparse_equiv
    if core_sparse_equiv(DICT1, DICT2) is False:
        return False

    # IF GET TO THIS POINT, MUST BE TRUE
    return True


def return_uniques(DICT1:dict) -> np.ndarray:

    """Return unique values of a sparse dictionary as numpy.ndarray.

    Parameters
    ----------
    DICT1:
        dict - object for which to get unique values

    Return
    ------
    UNIQUES:
        ndarray - Vector of unique values

    See Also
    --------
    numpy.unique

    Examples
    --------
    >>> from pybear.sparse_dict import return_uniques
    >>> SD = {0:{0:1, 1:8, 2:3}, 1:{0:8, 1:3, 2:7}}
    >>> out = return_uniques(SD)
    >>> out
    array([1, 3, 7, 8], dtype=object)

    """

    if DICT1 is None:
        raise TypeError(f"'DICT1' cannot be None")

    DICT1 = _dict_init(DICT1)
    ida1(DICT1)

    if _is_sparse_inner(DICT1):
        DICT1 = {0: DICT1}

    NUM_HOLDER, STR_HOLDER = [], []

    # 22_10_16 DO NOT CHANGE THIS, HAS TO DEAL W DIFF DTYPES
    # 24_05_02 dict_init PROHIBITS NON-NUMERICS IN SPARSE DICTIONARIES
    # DONT USE np.unique, BLOWS UP FOR '<' not supported between 'str' and 'int'
    for outer_key in DICT1:
        for value in DICT1[outer_key].values():
            if any(map(lambda x: x in str(type(value)).upper(), ['INT', 'FLOAT'])):
                if value not in NUM_HOLDER: NUM_HOLDER.append(value)
            else:
                if value not in STR_HOLDER: STR_HOLDER.append(str(value))

    if sparsity(DICT1) > 0 and 0 not in NUM_HOLDER: NUM_HOLDER.append(0)

    UNIQUES = np.array(sorted(NUM_HOLDER) + sorted(STR_HOLDER), dtype=object)

    del NUM_HOLDER, STR_HOLDER

    return UNIQUES



def drop_placeholders(DICT1:dict) -> dict:

    """Remove placeholding zeros from a properly constructed sparse dictionary.

    Parameters
    ----------
    DICT1:
        dict - The sparse dictionary object from which to remove place-holding
        zeros.

    Return
    ------
    DICT1:
        dict - Sparse dictionary with place-holding zeros removed.

    Examples
    --------
    >>> from pybear.sparse_dict import drop_placeholders
    >>> SD = {0: {0:1, 2:0}, 1: {2:2}, 2: {2:0}}
    >>> out = drop_placeholders(SD)
    >>> out
    {0: {0: 1}, 1: {2: 2}, 2: {}}

    """

    _was_inner = False
    if _is_sparse_inner(DICT1):
        DICT1 = {0: DICT1}
        _was_inner = True

    DICT1 = _dict_init(DICT1)

    last_inner_key = inner_len(DICT1) - 1
    for outer_key in DICT1:
        if last_inner_key in DICT1[outer_key]:
            if DICT1[outer_key][last_inner_key] == 0:
                del DICT1[outer_key][last_inner_key]

    if _was_inner:
        DICT1 = DICT1[0]

    return DICT1


def dtype_(DICT:dict):

    """
    Return the dtype of the values in a sparse dictionary. Returns a TypeError
    if there is more that one dtype.

    Parameters
    ----------
    DICT: dict - the dictionary for which to get dtype

    Return
    ------
    dtype:
        python or numpy dtype of the values within the sparse dictionary

    Examples
    --------
    >>> from pybear.sparse_dict import zip_array
    >>> from pybear.sparse_dict import dtype_
    >>> NP = np.random.randint(0, 10, (3, 3), dtype=np.uint8)
    >>> SD = zip_array(NP, dtype=np.uint8)
    >>> dtype_(SD)
    <class 'numpy.uint8'>

    """

    ida1(DICT)
    _dict_init(DICT)

    if _is_sparse_inner(DICT):
        DTYPES = np.unique(list(map(str, map(type, DICT.values()))))

        _dtype = type(list(DICT.values())[0])

    elif _is_sparse_outer(DICT):

        DTYPES = np.empty((0,))
        for outer_key in DICT:

            DTYPES = np.hstack((DTYPES,
                np.unique(list(map(str, map(type, DICT[outer_key].values()))))))

        DTYPES = np.unique(DTYPES)

        _dtype = type(list(DICT[list(DICT.keys())[0]].values())[0])

    if len(DTYPES) > 1:
        raise ValueError(f"DICT contains more than one dtype: "
                         f"{', '.join(DTYPES.tolist())}")
    else:
        del DTYPES
        return _dtype


def astype(DICT:dict, dtype=float) -> dict:

    """
    Set the dtype of the values in a sparse dictioanry.

    Parameters
    ----------
    DICT:
        dict - sparse dictionary for which to set dtype
    dtype:
        [int, float, np.int8, np.int16, np.int32, np.int64, np.uint8,
        np.uint16, np.uint32, np.uint64, np.float16, np.float32, np.float64],
        default = float - the dtypes of the values in the returned sparse
        dictionary.


    Returns
    -------
    DICT:
        dict - the passed sparse dictionary with new dtype

    Examples
    --------
    >>> from pybear import sparse_dict
    >>> SD = sparse_dict.random.randint(0, 10, (3,3), 50, dtype=np.uint8)
    >>> sparse_dict.dtype_(SD)
    <class 'numpy.uint8'>
    >>> SD = astype(SD, dtype=np.float64)
    >>> sparse_dict.dtype_(SD)
    <class 'numpy.float64'>

    """

    _dict_init(DICT)
    ida1(DICT)

    if _is_sparse_inner(DICT):
        DICT = dict((zip(DICT.keys(), list(map(dtype, DICT.values())))))

    elif _is_sparse_outer(DICT):
        for outer_key in DICT:
            DICT[int(outer_key)] = dict((zip(DICT[outer_key].keys(),
                                 list(map(dtype, DICT[outer_key].values())))))

    return DICT







