# Authors:
#
#       Bill Sousa
#
# License: BSD 3 clause



import numbers

import numpy as np
import joblib



def serial_index_mapper(
    shape:tuple,
    positions:[list,tuple],
    n_jobs:int=None
) -> list:
    """Map serial index positions to their zero-based Cartesian
    coordinates in an object of the given shape.

    Parameters
    ----------
    shape : tuple of integers
        The dimensions of the object to map into.
    positions : array_like of integers
        Vector of serialized index positions.
    n_jobs : int, default=None
        Number of CPU cores used when parallelizing over positions during
        mapping. None means 1 unless in a joblib.parallel_backend context.
        -1 means using all processors.

    Returns
    -------
    coordinates:
        list of tuples containing zero-based Cartesian
        coordinates for each given serialized index position.


    Example
    ------
    >>> from pybear.utilities import serial_index_mapper
    >>> shape = (3,3,3)
    >>> positions = [4, 15, 25]
    >>> coordinates = serial_index_mapper(shape, positions, n_jobs=1)
    >>> print(coordinates)
    [(0, 1, 1), (1, 2, 0), (2, 2, 1)]

    """

    # shape ** * ** * ** * ** * ** ** * ** * ** * ** * ** ** * ** * ** *
    err_msg = (f"'shape' must be non-empty one-dimensional array-like containing "
               f"non-negative integers")

    try:
        if isinstance(shape, (dict, str, type(None))):
            raise Exception
        shape = np.array(list(shape))
        if len(shape) == 0:
            raise UnicodeError
        if not np.array_equiv(shape, shape.ravel()):
            raise UnicodeError
        if not all(map(isinstance, shape, (numbers.Integral for i in shape))):
            raise UnicodeError
        if any([i < 0 for i in shape]):
            raise UnicodeError
    except UnicodeError:
        raise ValueError(err_msg)
    except Exception as e:
        raise TypeError(err_msg)


    del err_msg

    # END shape ** * ** * ** * ** * ** * ** * ** * ** * ** ** * ** * **

    # positions ** * ** * ** * ** * ** * ** * ** * ** * ** * ** ** * **
    err_msg = (f"'positions' must be non-empty one-dimensional array-like "
               f"containing integers")

    try:
        if isinstance(positions, (dict, str, type(None))):
            raise Exception
        positions = np.array(list(positions))
        if len(positions) == 0:
            raise UnicodeError
        if not np.array_equiv(positions, positions.ravel()):
            raise UnicodeError
        if not all(map(isinstance, positions, (numbers.Integral for i in positions))):
            raise UnicodeError
    except UnicodeError:
        raise ValueError(err_msg)
    except Exception as e:
        raise TypeError(err_msg)

    del err_msg


    if any(map(lambda x: x < 0 or x > np.prod(shape), positions)):
        raise ValueError(
            f"a serialized index position is out of bounds for an object "
            f"of size {np.prod(shape)}"
        )

    # END positions ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

    # n_jobs ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    err_msg = f"n_jobs must be an integer in range -1 to 32 but not 0"

    try:
        if n_jobs is None:
            n_jobs = 1
            raise MemoryError
        float(n_jobs)
        if not int(n_jobs) == n_jobs:
            raise UnicodeError
        n_jobs = int(n_jobs)
        if n_jobs == 0 or n_jobs < -1:
            raise UnicodeError
    except MemoryError:
        pass
    except UnicodeError:
        raise ValueError(err_msg)
    except Exception as e:
        raise TypeError(err_msg)

    # END n_jobs ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    def _recursive(_posn, _coordinates, ctr):

        if ctr == 100:
            raise RecursionError(f"Recursion depth has surpassed 100")

        if len(_coordinates) == len(shape) - 1:
            _coordinates.append(int(_posn))
            return tuple(_coordinates)
        else:
            # len(COORDINATE) is the axis we are looking to find next
            _axis = len(_coordinates)
            _remaining_axes = shape[_axis + 1:]
            _current_axis_posn = int(_posn // np.prod(_remaining_axes))
            _coordinates.append(_current_axis_posn)
            _positions_consumed = _current_axis_posn * np.prod(_remaining_axes)
            if _positions_consumed == 0:
                # POSN = POSN
                pass
            else:
                _posn = _posn % _positions_consumed

            ctr += 1
            return _recursive(_posn, _coordinates, ctr)


    # DONT HARD-CODE backend, ALLOW A CONTEXT MANAGER TO SET
    with joblib.parallel_config(prefer='processes', n_jobs=n_jobs):
        coordinates = joblib.Parallel(return_as='list')(
            joblib.delayed(_recursive)(POSN, [], 1) for POSN in positions
        )


    return coordinates







