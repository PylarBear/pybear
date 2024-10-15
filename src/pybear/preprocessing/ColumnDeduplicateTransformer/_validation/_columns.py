# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from .._type_aliases import ColumnsType, DataType

import numpy as np



def _val_columns(_columns: ColumnsType, _X:DataType) -> None:

    if _columns is None:
        return

    _err_msg = (f"'columns', if passed, must be a list-like of strings "
        f"with n_features unique entries. ")

    try:
        iter(_columns)
        if isinstance(_columns, (str, dict)):
            raise TypeError
        try:
            # this is not returned here!
            _columns = np.array(list(_columns)).ravel()
        except:
            raise TypeError
        if not all(map(isinstance, _columns, (str for _ in _columns))):
            raise ValueError
        if len(np.unique(_columns)) != len(_columns):
            raise ValueError
    except TypeError:
        raise TypeError(_err_msg)
    except ValueError:
        raise ValueError(_err_msg)
    except:
        raise Exception(
            f"'_val_columns' raised for exception other than TypeError "
            f"or ValueError"
        )


    _, __ = len(_columns), _X.shape[1]
    if _ != __:
        raise ValueError(_err_msg + f" X num features: {__}, len 'columns': {_}")




