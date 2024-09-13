import numpy as np
import sparse_dict as sd
import inspect, warnings



# GETS FORMAT AND DOUBLES UP OBJECT IF WAS SINGLE, I.E. [] -->
# [[]] OR {} --> {0: {}}


class IsSparseOuterOrInnerError(Exception):
    pass

class NotArrayOrSparseDictError(Exception):
    pass


def list_dict_validater(OBJECT, object_name_as_str):

    """
    Validate given object is "ARRAY", "SPARSE_DICT", or None; check
    for [[]] or {0:{}} and fix, and return.

    Parameters
    ----------
    OBJECT:
        Iterable[str] -
    object_name_as_str:
        str -

    Return
    ------
    -
        tuple[str, Iterable[Iterable[str]]] -

    """

    if OBJECT is None:

        return None, None

    elif isinstance(OBJECT, (list, tuple, np.ndarray)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            OBJECT = np.array(OBJECT)
            if len(OBJECT.shape)==1: OBJECT = OBJECT.reshape((1,-1))

        return 'ARRAY', OBJECT

    elif isinstance(OBJECT, dict):
        if sd.is_sparse_outer(OBJECT) is False and \
                sd.is_sparse_inner(OBJECT) is True:
            OBJECT = {0: OBJECT}
        elif sd.is_sparse_outer(OBJECT) is True and \
                sd.is_sparse_inner(OBJECT) is False:
            # ENSURE OUTER DICT KEYS ARE ZERO-BASED-CONTIGUOUS
            if not list(OBJECT.keys())==list(range(len(OBJECT))):
                print(f'\n *** list_dict_validater >>> OUTER KEYS WERE '
                      f'NOT ZERO-BASED CONTIGUOUS FOR SD OBJECT ***\n')
                __ = input(f'HIT ENTER TO CONTINUE > ')
                OBJECT = dict((zip(range(len(OBJECT)),OBJECT.values())))

        else: raise IsSparseOuterOrInnerError(
            f'{inspect.stack()[0][3]} >>> sd.is_sparse_outer and/or '
            f'sd.is_sparse_inner logic is failing'
        )

        return 'SPARSE_DICT', OBJECT

    else: raise NotArrayOrSparseDictError(
        f'INVALID type "{type(OBJECT)}" FOR {object_name_as_str}, MUST '
        f'BE LIST-TYPE, SPARSE_DICT, OR None'
    )








