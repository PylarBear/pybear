# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


# pizza
from typing import Iterable, Literal, Optional
from typing_extensions import Union, Self
from ._type_aliases import (
    KeepType, DataType
)
from numbers import Real, Integral


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils._param_validation import StrOptions

from ._partial_fit._partial_fit import _partial_fit
from ._partial_fit._merge_constants import _merge_constants
from ._partial_fit._make_instructions import _make_instructions




class InterceptManager(BaseEstimator, TransformerMixin):

    """
    pizza say something


    Parameters
    ----------
    keep:
        Optional[Union[Literal['first', 'last', 'random'], dict[str,any], None]],
        default='last' - pizza!
    equal_nan:
        Optional[bool], default=True - pizza!
    rtol:
        float, default = 1e-5 - The relative difference tolerance for
            equality. See numpy.allclose.
    atol:
        float, default = 1e-8 - The absolute tolerance parameter for .
            equality. See numpy.allclose.
    n_jobs:
        # pizza finalize this based on benchmarking
        Optional[Integral], default=-1 - The number of joblib Parallel
        jobs to use when scanning the data for columns of constants. The
        default is to use processes, but can be overridden externally
        using a joblib parallel_config context manager. The default
        number of jobs is -1 (all processors). To get maximum speed
        benefit, pybear recommends using the default setting.


    Attributes
    ----------
    constant_columns_:
        dict[int, any] - pizza!

    kept_columns_:
        dict[int, any] - pizza!

    removed_columns_:
        dict[int, any] - pizza!


    See Also
    --------
    numpy.ndarray
    pandas.core.frame.DataFrame
    scipy.sparse
    numpy.allclose


    """


    _parameter_constraints: dict = {
        "keep": [StrOptions({"first", "last", "random", "none"}), dict[str, any]],
        "equal_nan": ["boolean"],
        "rtol": [Real],
        "atol": [Real],
        "n_jobs": [Integral, None]
    }


    def __init__(
        self,
        keep: KeepType='last',
        equal_nan: Optional[bool]=True,
        rtol: Optional[Real]=1e-5,
        atol: Optional[Real]=1e-8,
        n_jobs: Optional[Integral]=-1  # pizza benchmark what is the best setting
    ):

        self.keep = keep
        self.equal_nan = equal_nan
        self.rtol = rtol
        self.atol = atol
        self.n_jobs = n_jobs



    def _reset(self):

        """
        Reset internal data-dependent state of the transformer.
        __init__ parameters are not changed.

        """

        if hasattr(self, 'constant_columns_'):
            del self.constant_columns_
            del self.kept_columns_
            del self.removed_columns_


    def partial_fit(
        self,
        X:DataType,
        y:any=None
    ) -> Self:

        """
        pizza say words


        Parameters
        ----------


        Return
        ------
        -
            self: the fitted InterceptManager instance




        """


        # pizza remember to set order to "F"!

        # dictionary of column indices and respective constant values
        _new_constants:dict[int, any] = \
            _partial_fit(
                X,
                self.equal_nan,
                self.rtol,
                self.atol,
                self.n_jobs
            )

        # merge newly found constant columns with those found during
        # previous partial fits
        self.constant_columns_ = _merge_constants(
            self.constant_columns_ if hasattr(self, 'constant_columns_') else {},
            _new_constants
        )

        _instructions = _make_instructions(self.keep, self.constant_columns_)

        self.kept_columns_: dict[int, any] = {}
        self.removed_columns_: dict[int, any] = {}
        for col_idx in range(X.shape[1]):
            if col_idx in _instructions['keep'] or {}:
                self.kept_columns_[col_idx] = self.constant_columns_[col_idx]
            elif col_idx in _instructions['add'] or {}:
                self.kept_columns_[col_idx] = self.constant_columns_[col_idx]
            elif col_idx in _instructions['delete'] or {}:
                self.removed_columns_[col_idx] = self.constant_columns_[col_idx]



        return self


    def fit(
        self,
        X:DataType,
        y:any=None
    ) -> Self:

        """
        pizza say words


        Parameters
        ----------
        X:


        y:


        Return
        ------
        -
            self: the fitted InterceptManager instance



        """

        # pizza remember to set order to "F"!


        return self.partial_fit(X)


    def transform(
        self,
        X: DataType,
        copy: bool=None
    ) -> DataType:
        """
        pizza say words


        Parameters
        ----------


        Return
        ------




        """

        # pizza remember to set order to "F"!

        # this needs to be redone every transform in case 'keep' was
        # changed via set params
        _instructions = _make_instructions(self.keep, self.constant_columns_)

        self.kept_columns_: dict[int, any] = {}
        self.removed_columns_: dict[int, any] = {}
        for col_idx in range(X.shape[1]):
            if col_idx in _instructions['keep'] or {}:
                self.kept_columns_[col_idx] = self.constant_columns_[col_idx]
            elif col_idx in _instructions['add'] or {}:
                self.kept_columns_[col_idx] = self.constant_columns_[col_idx]
            elif col_idx in _instructions['delete'] or {}:
                self.removed_columns_[col_idx] = self.constant_columns_[col_idx]

        return _transform(X, _instructions)


    def inverse_transform(
        self,
        X: DataType
    ) -> DataType:

        """
        pizza say words


        Parameters
        ----------


        Return
        ------




        """

        # pizza remember to set order to "F"!


        return












