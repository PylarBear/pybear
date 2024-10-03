# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from .._type_aliases import ColumnsType, DataType




def _val_columns(_columns: ColumnsType, _X:DataType) -> None:

    _err_msg = (f"'columns', if passed, must be a list-like of strings"
        f"with n_features entries. ")

    if _columns is None:
        pass
    else:
        try:
            iter(_columns)
            if isinstance(_columns, (str, dict)):
                raise TypeError
            try:
                _columns = _columns.ravel()
            except:
                pass
            if not all(map(isinstance, _columns, (str for _ in _columns))):
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


    if _columns is not None and len(_columns) != _X.shape[1]:
        raise ValueError(_err_msg)







