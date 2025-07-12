# Authors:
#
#       Bill Sousa
#
# License: BSD 3 clause
#



from typing import (
    Iterable,
    Literal,
    Optional
)
from typing_extensions import (
    Any,
    Union
)
import numpy.typing as npt

import numbers

import joblib
from numpy import (
    array, ceil, empty, hstack, int8, int32, prod, random as np_random,
    sum, zeros
)

from pybear.utilities._serial_index_mapper import serial_index_mapper as sim
from pybear.utilities._array_sparsity import array_sparsity as arsp



def choice(
    a: Iterable[Any],
    shape: Union[int, Iterable[int]],
    replace: Optional[bool]=True,
    n_jobs: Optional[Union[int, None]]=None
) -> npt.NDArray[Any]:

    """
    Select math.prod(shape) quantity of elements from the given pool "a",
    with or without replacement. This module improves on the impossible
    slowness of numpy.random.choice on large "a" when replace=False.
    Enter "a" as a 1-dimensional vector. A "p" argument is not available
    as this algorithm as it relies on the assumption of equal likelihood
    for all values in "a".


    Parameters
    ----------
    a:
        Iterable[Any] - 1-dimensional list-like of elements to randomly
        choose from.
    shape:
        Union[int, Iterable[int]] - Shape of returned numpy array
        containing the randomly selected values.
    replace:
        bool - Select values from 'a' with (True) or without (False)
        replacement of previous pick.
    n_jobs:
        Union[int, None], default=None - Number of CPU cores used
        when parallelizing over subpartitions of 'a' during selection.
        -1 means using all processors.


    Returns
    -------
    -
        PICKED:
            NPArray[Any] - elements selected from 'a' of shape 'shape'.


    See Also
    --------
    numpy.random.choice


    Examples
    --------
    >>> from pybear.new_numpy import random as pb_random
    >>> result = pb_random.choice(list(range(20)), (3,2), n_jobs=1)
    >>> print(result) #doctest:+SKIP
    [[ 9  6]
     [ 2  0]
     [11  8]]

    """


    err_msg = (f"'a' must be a non-empty 1-dimensional array-like that "
               f"can be converted to a numpy array")

    if isinstance(a, (str, dict)):
        raise TypeError(err_msg)

    try:
        list(a[:10])
        a = array(a)
    except:
        raise TypeError(err_msg)


    if len(a.shape) != 1:
        raise ValueError(err_msg)

    if len(a) == 0:
        raise ValueError(err_msg)

    del err_msg


    err_msg = f"'shape' must be an integer or a tuple"
    # shape can be > 2 dimensional
    try:
        float(shape)
        if not int(shape)==shape:
            raise TypeError(err_msg)
        shape = (int(shape),)
    except:
        try:
            list(shape)
            shape = tuple(shape)
        except:
            raise TypeError(err_msg)

    del err_msg


    if not isinstance(replace, bool):
        raise TypeError(f"'replace' kwarg must be boolean")

    pick_qty = prod(shape)

    if replace is False and pick_qty > a.size:
        raise ValueError(f'quantity of selected cannot be greater '
                         f'than pool size when replace=False')


    # n_jobs ** * ** * ** * ** * ** * ** * ** *
    if n_jobs is not None:
        err_msg = \
            f"n_jobs must be a positive integer, -1, or None, got '{n_jobs}'"
        try:
            float(n_jobs)
            if int(n_jobs) != n_jobs:
                raise UnicodeError
            n_jobs = int(n_jobs)
            if n_jobs < 1 and n_jobs != -1:
                raise UnicodeError
        except UnicodeError:
            raise ValueError(err_msg)
        except Exception as e:
            raise TypeError(err_msg)
    # END n_jobs ** * ** * ** * ** * ** * ** * ** *


    partition_size = min(a.size, int(2**16))

    psis = range(0, a.size, partition_size)  # partition_start_indices

    @joblib.wrap_non_picklable_objects
    def _puller(subpartition_of_a, _size, pick_qty, replace):

        PULL = np_random.choice(
            subpartition_of_a,
            int(ceil(len(subpartition_of_a) / _size * pick_qty)),
            replace=replace
        )
        return PULL

    ARGS = [a.size, pick_qty, replace]

    # 'a' MUST BE 1-D
    with joblib.parallel_config(prefer='processes', n_jobs=n_jobs):
        PULLED = joblib.Parallel(return_as='list')(
            joblib.delayed(_puller)(a[psi:psi+partition_size], *ARGS) for psi in psis
        )

    PICKED = hstack((PULLED))

    del partition_size, psis, _puller, ARGS, PULLED


    if PICKED.size > pick_qty:
        PICKED = np_random.choice(PICKED, pick_qty, replace=False)
    elif PICKED.size < pick_qty:
        raise AssertionError(
            f"'PICKED' is smaller than pick_qty, algorithm failure"
        )

    return PICKED.reshape(shape)



class Sparse:

    """
    Return random values from a “discrete uniform” (integer) or "uniform"
    (float) distribution of the specified dtype in the “half-open”
    interval [low, high) (includes low, but excludes high), with desired
    sparsity.

    Samples are uniformly distributed over the half-open interval [low,
    high) . In other words, any value within the given interval is
    equally likely to be drawn.


    Parameters
    ----------
    minimum:
        numbers.Real - Lowest (signed) value to be drawn from the
        distribution.
    maximum:
        numbers.Real - Upper boundary of the output interval. All values
        generated will be less than this number.
    shape:
        Union[int, Iterable[int]] - Dimensions of the returned array.
    sparsity:
        numbers.Real, default = 0 - Desired percentage of zeros in
        the returned array.
    engine:
        Literal["choice", "filter", "serialized", "iterative", "default"],
        default = "default" -  Selects the desired engine for generating
        the returned array. Some engines offer higher speed with lower
        accuracy, while others have higher accuracy at the expense of
        speed. "default" behavior is a hybrid of "filter" and "iterative".

        "choice" - Build a full-size mask with sparse locations
        determined by numpy.random.choice on [0,1], with p achieving
        amount of sparsity. Apply the mask to a full-sized 100% dense
        numpy.ndarray filled as dictated by parameters to populate it
        with zeros.

        "filter" - Generate an array filled randomly from [1,100000] and
        convert the array to a mask that fixes the sparse locations by
        applying a number filter derived from the tartet sparsity.
        Generate a 100% dense array of ints or floats then apply the mask
        to it to achieve sparsity.

        "serialized" - Generate a serialized list of unique indices and
        random values (or zeros) then map the values (or zeros) into a
        fully sparse (or dense) array.

        i) Deterimine the number of dense (or sparse) positions in
            the target array.
        ii) Generate that number of random dense (or sparse) indices
            serially using pybear.new_numpy.random.choice *without
            replacement*. This guarantees no duplicate indices.
        iii) Generate an equally-sized vector of dense values (or zeros).
        iv) Map the vector of values (or zeros) to the index
            positions in a 100% sparse (or dense) full-sized array.

        "iterative" - Generate a serialized list of not-necessarily-
        unique indices and random values (or zeros), then map the values
        (or zeros) into a fully sparse (or dense) array. Repeat
        iteratively until the desired sparsity is achieved. Same as
        _serialized except these indices are not necessarily unique and
        the process is iterative.

        i) Determine the number of dense (or sparse) positions in
            the target array.
        ii) Generate that number of random dense (or sparse) indices
            serially *with replacement*. This does not guarantee
            non-duplicate indices.
        iii) Generate an equally-sized vector of values (or zeros).
        iv) Map the vector of values (or zeros) to the index positions
            in a 100% sparse (or dense) full-sized array.
        v) Because there may have been duplicate indices, repeat steps
            ii-iv until desired sparsity is achieved.

        "default" - A hybrid method of "filter" and "iterative" that
        maximizes speed and accuracy. When the size of the target object
        is less than 1,000,000, the fastest methods "filter" and "choice"
        have difficulty achieving the target sparsity. In this case, the
        more accurate, but slower, "iterative" method is used. For target
        sizes over 1,000,000, the law of averages prevails and the
        "filter" method is able to achieve sufficiently close sparsities
        at speeds much faster than "iterative".
    dtype:
        Any, default = float - Desired dtype of the result.


    Attributes
    ----------
    sparse_array_:
        ndarray of shape 'shape'.


    See Also
    --------
    numpy.random.randint
    numpy.random.uniform


    Examples
    --------
    >>> from pybear.new_numpy import random as pb_random
    >>> instance = pb_random.Sparse(0, 10, (3,3), 50, dtype=int8)
    >>> sparse_array = instance.sparse_array_
    >>> print(sparse_array) #doctest:+SKIP
    [[0 6 0]
     [8 8 0]
     [0 0 1]]


    """

    def __init__(
        self,
        minimum: numbers.Real,
        maximum: numbers.Real,
        shape: Union[int, Iterable[int]],
        sparsity: numbers.Real = 0,
        engine: Literal['choice', 'filter', 'serialized', 'iterative', 'default'] = \
            'default',
        dtype: Optional[Any] = float,
    ):

        self._min = minimum
        self._max = maximum
        self._shape = shape
        self._sparsity = sparsity
        self._engine = engine
        self._dtype = dtype

        # VALIDATION ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

        # dtype ** * ** * ** * ** * ** * ** * ** *
        # THIS MUST BE BEFORE _min & _max
        if not isinstance(self._dtype, type(int)):
            raise TypeError(f'dtype must be a valid py or numpy dtype')
        # dtype ** * ** * ** * ** * ** * ** * ** *


        # _min ** * ** * ** * ** * ** * ** *
        try:
            float(self._min)
        except:
            raise TypeError(f"'minimum' must be numeric")

        if 'INT' in str(self._dtype).upper():
            if int(self._min) != self._min:
                raise ValueError(
                    f"'minimum' must be an integer when dtype is integer"
                )
        elif 'FLOAT' in str(self._dtype).upper():
            if self._min == float('inf') or self._min == float('-inf'):
                raise ValueError(f"'minimum' cannot be infinity")
        # END _min ** * ** * ** * ** * ** * ** *

        # _max ** * ** * ** * ** * ** * ** *
        try:
            float(self._max)
        except:
            raise TypeError(f"'minimum' must be numeric")

        if 'INT' in str(self._dtype).upper():
            if int(self._max) != self._max:
                raise ValueError(
                    f"'maximum' must be an integer when dtype is integer"
                )
        elif 'FLOAT' in str(self._dtype).upper():
            if self._max == float('inf') or self._max == float('-inf'):
                raise ValueError(f"'maximum' cannot be infinity")
        # END _max ** * ** * ** * ** * ** * ** *

        # _min v _max ** * ** * ** * ** * ** * ** *
        if 'INT' in str(self._dtype).upper():
            if self._min >= self._max:
                raise ValueError(f"when dtype is integer, 'minimum' must be "
                    f"<= ('maximum' - 1)")
        elif 'FLOAT' in str(self._dtype).upper():
            if self._min > self._max:
                self._min, self._max = self._max, self._min
        # END _min v _max ** * ** * ** * ** * ** * ** *


        # shape ** * ** * ** * ** * ** * ** * ** *

        if isinstance(self._shape, (str, dict)):
            raise TypeError(f"'shape' expected a sequence of integers or a "
                f"single integer, got type '{type(self._shape)}'")
        elif isinstance(self._shape, type(None)):
            self._shape = ()
        else:
            try:
                tuple(self._shape)
            except:

                err_msg = (f"'shape' expected a sequence of integers or a "
                           f"single integer, got '{self._shape}'")
                try:
                    float(self._shape)
                except:
                    raise TypeError(err_msg)

                if int(self._shape) != self._shape:
                    raise TypeError(err_msg)
                if self._shape < 0:
                    raise ValueError(f"negative dimensions are not allowed")
                self._shape = (self._shape,)

                del err_msg


            if len(array(self._shape).shape) > 1:
                raise TypeError(f"'shape' expected a sequence of integers or "
                    f"a single integer")

            self._shape = tuple(self._shape)

            err_msg = f"cannot be interpreted as an integer"
            for item in self._shape:
                try:
                    float(item)
                except:
                    raise TypeError(f"'shape' {type(item)} {err_msg}")

                if int(item) != item:
                    raise TypeError(f"{item} {err_msg}")
                if item < 0:
                    raise ValueError(f"negative dimensions are not allowed")

            del err_msg


        # END shape ** * ** * ** * ** * ** * ** * ** *


        # sparsity ** * ** * ** * ** * ** * ** * ** *
        err_msg = f"sparsity must be a number between 0 and 100, inclusive"
        try:
            float(self._sparsity)
        except:
            raise TypeError(err_msg)

        if self._sparsity < 0 or self._sparsity > 100:
            raise ValueError(err_msg)

        if 'INT' in str(self._dtype).upper():
            if self._min==0 and self._max==1 and self._sparsity != 100:
                raise ValueError(f"cannot satisfy the impossible condition of "
                    f"'minimum' = 0 'maximum' = 1 and 'sparsity' != 100 for "
                    f"integer dtype")
        elif 'FLOAT' in str(self._dtype).upper():
            if self._min==0 and self._max==0 and self._sparsity != 100:
                raise ValueError(f"cannot satisfy the impossible condition of "
                    f"'minimum' = 0 'maximum' = 0 and 'sparsity' != 100 for "
                    f"float dtype")

        del err_msg
        # END sparsity ** * ** * ** * ** * ** * ** * **


        # engine ** * ** * ** * ** * ** * ** * ** * ** * **

        allowed = ['choice', 'filter', 'serialized', 'iterative', 'default']
        err_msg = (f"'engine' must be {', '.join(allowed)}")

        if not isinstance(self._engine, str):
            raise TypeError(err_msg)

        self._engine = self._engine.lower()

        if self._engine not in allowed:
            raise ValueError(
                f"'{self._engine}' is not in allowed, must be "
                f"{', '.join(map(str, allowed))}"
            )
        del allowed

        # END engine ** * ** * ** * ** * ** * ** * ** * ** *

        # END VALIDATION ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


        if self._shape == 0:
            self.sparse_array_ = array([], dtype=self._dtype)

        try:
            if 0 in self._shape:
                self.sparse_array_ = \
                    array([], dtype=self._dtype).reshape(self._shape)
        except:
            pass


        if self._sparsity == 0:
            self.sparse_array_ = self._make_base_array_with_no_zeros(
                self._min,
                self._max,
                self._shape,
                self._dtype
            ).astype(self._dtype)

        if self._sparsity == 100:
            self.sparse_array_ = zeros(self._shape, dtype=self._dtype)

        if self._engine == "choice":
            self.sparse_array_ = self._choice()

        elif self._engine == "filter":
            self.sparse_array_ = self._filter()

        elif self._engine == "serialized":
            self.sparse_array_ = self._serialized()

        elif self._engine == "iterative":
            self.sparse_array_ = self._iterative()

        elif self._engine == "default":

            # IF total_size IS ABOVE 1e6, MAKE BY FILTER METHOD, IS MUCH FASTER
            # THAN SERIALIZED OR ITERATIVE AND LAW OF AVERAGES SHOULD GET
            # SPARSITY CLOSE ENOUGH. BUT WHEN SIZE IS SMALL, "FILTER" AND
            # "CHOICE" HAVE A HARD TIME GETTING SPARSITY CLOSE ENOUGH, SO USE
            # ITERATIVE.
            if prod(self._shape) >= 1e6:
                self.sparse_array_ = self._filter()
            else:
                self.sparse_array_ = self._iterative()
        else:
            raise AssertionError(f"logic managing engine selection failed")


    def _make_base_array_with_no_zeros(
        self, _min, _max, _shape, _dtype
    ) -> npt.NDArray[numbers.Real]:

        """
        Generate an array based on the given minimum, maximum, shape, and
        dtype rules, and iteratively replace any zeros with non-zero
        values generated by the same rules.


        Parameters
        ----------
        _min:
            int - the low end of the range of random numbers in the
                fully-dense base array
        _max:
            int - the hign end of the range of random numbers in the
                fully-dense base array
        _shape:
            tuple[int] - the shape of the final sparse array
        _dtype:
            the dtype of the final sparse array


        Returns
        -------
        -
            BASE_ARRAY: npt.NDArray[numbers.Real] - fully dense
                numpy ndarray with minumum, maximum, shape, and dtype as
                specified.

        """

        # Set the numpy array generators to be used based on dtype.

        if 'INT' in str(self._dtype).upper():
            array_generator = np_random.randint

        elif 'FLOAT' in str(self._dtype).upper():
            # CREATE A WRAPPER FOR np.random.uniform SO THAT IT'S SIGNATURE IS
            # THE SAME AS np.random.randint. dtype WILL JUST PASS THROUGH.

            def new_rand_uniform(_min, _max, _shape, _dtype):
                return np_random.uniform(_min, _max, _shape)

            array_generator = new_rand_uniform


        def _make_patch(__shape):

            nonlocal _min, _max, _dtype

            return array_generator(_min, _max, __shape, _dtype)


        BASE_ARRAY = _make_patch(_shape)

        # DONT LOOP THE ZEROES OUT OF BASE_ARRAY "NON_ZERO_VALUES"!
        # LOOP THEM OUT OF THE PATCH!
        if 0 in BASE_ARRAY:
            MASK = (BASE_ARRAY == 0)
            PATCH = _make_patch(sum(MASK))

            while 0 in PATCH:
                PATCH_MASK = (PATCH == 0)
                PATCH[PATCH_MASK] = _make_patch(sum(PATCH_MASK))
                del PATCH_MASK

            BASE_ARRAY[MASK] = PATCH

            del MASK, PATCH

        del _make_patch

        return BASE_ARRAY


    def _calc_support_info(self):
        """Calculate supporting info for performing operations."""

        if len(self._shape)==0:
            self._total_size = 0
        else:
            self._total_size = prod(self._shape).astype(int32)

        self._dense_size = self._total_size / 100 * (100 - self._sparsity)

        # IF SPARSITY DOESNT GO EVENLY INTO NUM ELEMENTS (I.E. _dense_size IS
        # NOT AN INTEGER), RANDOMLY ROUND OFF _dense_size
        if self._dense_size % 1 > 0:
            self._dense_size = int(self._dense_size // 1 + np_random.randint(2))
        else:
            self._dense_size = int(self._dense_size)

        self._sparse_size = int(self._total_size - self._dense_size)

        try:
            self._target_sparsity = \
                round(100 * self._sparse_size / self._total_size, 12)
        except:
            self._target_sparsity = self._sparsity


    def _choice(self):
        """Apply a mask of bools generated by random.choice to a 100%
        dense array to achieve sparsity.

        Returns
        -------
        -
            SPARSE_ARRAY:
                npt.NDArray[numbers.Real] - sparse array of
                specified minimum & maximum (excluding zeros), shape and
                dtype, made by applying a random.choice binary mask to a
                fully dense array.

        """

        #######################################################################
        # METHOD 1 "choice" - BUILD A FULL-SIZED MASK WITH SPARSE LOCATIONS
        # DETERMINED BY random.choice ON [0,1], WITH p ACHIEVING AMOUNT OF
        # SPARSITY. APPLY MASK TO A FULL SIZED 100% DENSE NP ARRAY
        # FILLED AS DICTATED BY PARAMETERS.
        #######################################################################

        # REMEMBER! THE MASK IS GOING TO BE A BOOL TO REPRESENT PLACES IN THE
        # BASE ARRAY THAT WILL GO TO ZERO!  THAT MEANS THAT THE PLACES THAT
        # WILL BE ZERO MUST BE A ONE IN THE MASK, AND ZERO IF NOT GOING TO BE
        # ZERO! MAKE SENSE?
        MASK = np_random.choice(
            [1, 0],
            self._shape,
            replace=True,
            p=(
                self._sparsity / 100,
                (100 - self._sparsity) / 100
            )
        ).astype(bool)

        SPARSE_ARRAY = self._make_base_array_with_no_zeros(
            self._min,
            self._max,
            self._shape,
            self._dtype
        )

        SPARSE_ARRAY[MASK] = 0
        del MASK

        return SPARSE_ARRAY.astype(self._dtype)

        # END METHOD 1  #######################################################
        #######################################################################


    def _filter(self):
        """
        Generate an array filled randomly from [1,100000] and turn it into a
        mask by applying a number filter. Generate a 100% dense array of
        ints or floats then apply the mask to it to achieve sparsity.


        Returns
        -------
        SPARSE_ARRAY : npt.NDArray[numbers.Real]
            Sparse array of specified minimum & maximum (excluding zeros),
            shape and dtype, made by applying a number filter mask to a
            fully dense array.

        """

        #######################################################################
        # METHOD 2 "filter" - BUILD A FULL-SIZED ARRAY FILLED RANDOMLY ON RANGE
        # [0-100000]. CONVERT THE ARRAY TO A MASK THAT FIXES THE SPARSE LOCATIONS
        # BY APPLYING A NUMBER FILTER DERIVED FROM THE TARGET SPARSITY. APPLY
        # THE MASK OVER A FULL SIZED 100% DENSE NP ARRAY.
        #######################################################################

        # USE THIS TO DETERMINE WHAT WILL BECOME ZEROS
        MASK = np_random.randint(0, 100000, self._shape, dtype=int32)
        cutoff = (1 - self._sparsity / 100) * 100000
        MASK = (MASK >= cutoff).astype(bool)

        SPARSE_ARRAY = self._make_base_array_with_no_zeros(
            self._min,
            self._max,
            self._shape,
            self._dtype
        )
        SPARSE_ARRAY[MASK] = 0

        del MASK, cutoff

        return SPARSE_ARRAY.astype(self._dtype)

        # END METHOD 2 ########################################################
        #######################################################################


    def _serialized(self):
        """
        Generate a serialized list of unique indices and random values
        (or zeros) then map the values (or zeros) into a fully sparse
        (or dense) array.

        Returns
        -------
        SPARSE_ARRAY : npt.NDArray[numbers.Real]
            Sparse array of specified minimum & maximum (excluding zeros),
            shape and dtype, made by building a vector of positions that
            identify dense or sparse locations in the final array.

        """

        #######################################################################
        # METHOD 3 "serialized" -
        # i) DETERMINE THE NUMBER OF DENSE (OR SPARSE) POSITIONS IN THE ARRAY.
        # ii) GENERATE THAT NUMBER OF RANDOM DENSE (OR SPARSE) INDICES SERIALLY
        # USING random.choice *WITHOUT REPLACEMENT*. THIS GUARANTEES NO
        # DUPLICATE INDICES.
        # iii) GENERATE AN EQUALLY-SIZED VECTOR OF DENSE VALUES (OR ZEROS).
        # iv) MAP THE VECTOR OF VALUES (OR ZEROS) TO THE INDEX POSITIONS IN A
        # 100% SPARSE (OR DENSE) FULL SIZED ARRAY
        #######################################################################

        self._calc_support_info()

        # ALLOW pybear.new_numpy.random.choice TO SELECT FROM THE SMALLER OF
        # dense_size OR sparse_size, SAVES MEMORY & TIME

        # WHEN DENSE IS SMALLER OR _sparse_size == 0
        if self._dense_size == 0 and self._sparse_size == 0:
            return self._filter()

        elif (self._sparse_size >= self._dense_size > 0) \
            or self._sparse_size == 0:

            SERIAL_DENSE_POSNS = choice(
                range(self._total_size),
                self._dense_size,
                replace=False,
                n_jobs=-1
            ).astype(int32)

            SERIAL_DENSE_POSNS.sort()

            # CREATE RANDOM VALUES MATCHING THE DENSE SIZE
            SERIAL_VALUES = self._make_base_array_with_no_zeros(
                self._min,
                self._max,
                self._dense_size,
                self._dtype
            )

            SPARSE_ARRAY = zeros(self._shape, dtype=self._dtype)

            MAPPED_INDICES = sim(self._shape, SERIAL_DENSE_POSNS)

            SPARSE_ARRAY[tuple(zip(*MAPPED_INDICES))] = SERIAL_VALUES

            del SERIAL_DENSE_POSNS, SERIAL_VALUES, MAPPED_INDICES

            return SPARSE_ARRAY.astype(self._dtype)


        # WHEN SPARSE IS SMALLER OR _dense_size == 0
        elif (0 < self._sparse_size < self._dense_size) or self._dense_size == 0:

            if self._sparse_size:
                SERIAL_SPARSE_POSNS = choice(
                    range(self._total_size),
                    self._sparse_size,
                    replace=False,
                    n_jobs=-1
                ).astype(int32)

            SERIAL_SPARSE_POSNS.sort()

            SPARSE_ARRAY = self._make_base_array_with_no_zeros(
                self._min,
                self._max,
                self._shape,
                self._dtype
            )

            MAPPED_INDICES = sim(self._shape, SERIAL_SPARSE_POSNS)

            SPARSE_ARRAY[tuple(zip(*MAPPED_INDICES))] = 0

            del SERIAL_SPARSE_POSNS, MAPPED_INDICES

            return SPARSE_ARRAY.astype(self._dtype)


        # END METHOD 3 ########################################################
        #######################################################################


    def _iterative(self):
        """Generate a serialized list of not-necessarily-unique indices
        and random values (or zeros) then map the values (or zeros) into
        a fully sparse (or dense) array, and repeat iteratively until the
        desired sparsity is achieved. Same as _serialized except these
        indices are not necessarily unique and the process is iterative.

        Returns
        -------
        SPARSE_ARRAY : npt.NDArray[numbers.Real]
            Sparse array of specified minimum & maximum (excluding zeros),
            shape and dtype, made by building a vector of positions that
            identify dense or sparse locations in the final array.

        """

        #######################################################################
        # METHOD 4 - "iterative"
        # i) DETERMINE THE NUMBER OF DENSE (OR SPARSE) POSITIONS IN THE ARRAY.
        # ii) GENERATE THAT NUMBER OF RANDOM DENSE (OR SPARSE) INDICES SERIALLY
        # *WITH REPLACEMENT*. THIS DOES NOT GUARANTEE NON-DUPLICATE INDICES.
        # iii) GENERATE AN EQUALLY-SIZED VECTOR OF VALUES (OR ZEROS).
        # iv) MAP THE VECTOR OF VALUES (OR ZEROS) TO THE INDEX POSITIONS IN A
        # 100% SPARSE (OR DENSE) FULL SIZED ARRAY
        # v) BECAUSE THERE MAY HAVE BEEN DUPLICATE INDICES, REPEAT STEPS
        # ii - iv UNTIL DESIRED SPARSITY IS ACHIEVED
        #######################################################################

        self._calc_support_info()

        # ALLOW pybear.new_numpy.random.choice TO SELECT FROM THE SMALLER OF
        # dense_size OR sparse_size, SAVES MEMORY & TIME

        if self._dense_size == 0 and self._sparse_size == 0:

            return self._filter()

        elif self._sparse_size >= self._dense_size:  # WHEN DENSE IS SMALLER

            SPARSE_ARRAY = zeros(self._shape, dtype=self._dtype)

            _last_sparsity = 100

            # MAKE A RANDOM GRID OF COORDINATES
            while _last_sparsity != self._target_sparsity:
                need_dense_size = sum(SPARSE_ARRAY == 0) - self._sparse_size
                SERIAL_DENSE_POSNS = empty(
                    (need_dense_size, len(self._shape)),
                    dtype=int32
                )
                for _dim in range(len(self._shape)):
                    SERIAL_DENSE_POSNS[:, _dim] = \
                         np_random.randint(
                            0,
                            self._shape[_dim],
                            need_dense_size,
                            dtype=int32
                    )

                # CREATE RANDOM VALUES MATCHING THE DENSE SIZE
                SERIAL_VALUES = self._make_base_array_with_no_zeros(
                    self._min,
                    self._max,
                    need_dense_size,
                    self._dtype
                )

                SPARSE_ARRAY[tuple(zip(*SERIAL_DENSE_POSNS))] = SERIAL_VALUES

                _new_sparsity = round(arsp(SPARSE_ARRAY), 12)
                if _new_sparsity == self._target_sparsity:
                    break
                else:
                    _last_sparsity = _new_sparsity

            return SPARSE_ARRAY.astype(self._dtype)

        elif self._sparse_size < self._dense_size:  # WHEN SPARSE IS SMALLER

            SPARSE_ARRAY = self._make_base_array_with_no_zeros(
                self._min,
                self._max,
                self._shape,
                self._dtype
            )

            _last_sparsity = 0

            # MAKE A RANDOM GRID OF COORDINATES
            while _last_sparsity != self._target_sparsity:
                need_sparse_size = self._sparse_size - sum(SPARSE_ARRAY == 0)
                SERIAL_SPARSE_POSNS = empty(
                    (need_sparse_size, len(self._shape)),
                    dtype=int32
                )
                for _dim in range(len(self._shape)):
                    SERIAL_SPARSE_POSNS[:, _dim] = \
                        np_random.randint(0,
                            self._shape[_dim],
                            need_sparse_size,
                            dtype=int32
                    )

                SPARSE_ARRAY[tuple(zip(*SERIAL_SPARSE_POSNS))] = 0

                _new_sparsity = round(arsp(SPARSE_ARRAY), 12)
                if _new_sparsity == self._target_sparsity:
                    break
                else:
                    _last_sparsity = _new_sparsity

            return SPARSE_ARRAY.astype(self._dtype)

        # END METHOD 4 ########################################################
        #######################################################################



def sparse(
    minimum: numbers.Real,
    maximum: numbers.Real,
    shape: Union[int, Iterable[int]],
    sparsity: numbers.Real,
    dtype: Optional[Any] = float
):

    """
    Return random values from a “discrete uniform” (integer) or "uniform"
    (float) distribution of the specified dtype in the “half-open”
    interval [low, high), with desired sparsity.

    Samples are uniformly distributed over the half-open interval [low,
    high) (includes low, but excludes high). In other words, any value
    within the given interval is equally likely to be drawn.


    Parameters
    --------
    minimum:
        numbers.Real - Lowest (signed) value to be drawn from the
        distribution.
    maximum:
        numbers.Real - Upper boundary of the output interval. All values
        generated will be less than high.
    shape:
        Union[int, Iterable[int]] - Dimensions of the returned array.
    sparsity:
        Optional[numbers.Real], default = 0 - Desired percentage of
        zeros in the the returned array.
    dtype:
        Optional[Any], default = float - Desired dtype of the result.


    Returns
    -------
    -
        SPARSE_ARRAY: ndarray - array of dimensions 'shape' with random
            values from the appropriate distribution and with the
            specified sparsity.


    See Also
    -------
    numpy.random.randint
    numpy.random.uniform
    pybear.random.Sparse


    Examples
    -------
    >>> from pybear.new_numpy import random as pb_random
    >>> sparse_array = pb_random.sparse(11, 20, (4,4), 70, dtype=int8)
    >>> print(sparse_array)   #doctest:+SKIP
    [12  0  0 13]
    [0 16  0  0]
    [0  0  0 17]
    [0  0  0 16]]

    """

    return Sparse(
        minimum, maximum, shape, sparsity, "default", dtype
    ).sparse_array_





















