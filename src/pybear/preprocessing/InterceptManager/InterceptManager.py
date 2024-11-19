# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


# pizza dont forget to clean up these imports!!
from typing import Iterable, Literal, Optional
from typing_extensions import Union, Self
from ._type_aliases import (
    KeepType, DataType
)
from numbers import Real, Integral


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils._param_validation import StrOptions

from ._validation._validation import _validation

from ._partial_fit._find_constants import _find_constants
from ._partial_fit._make_instructions import _make_instructions
from ._partial_fit._set_attributes import _set_attributes

from ._transform._transform import _transform


class InterceptManager(BaseEstimator, TransformerMixin):

    """
    pizza say something


    Parameters
    ----------
    keep:
        Optional[Union[Literal['first', 'last', 'random'], dict[str,any], None]], int, str, callable]
        default='last' - pizza!
    equal_nan:
        Optional[bool], default=True - pizza!
    rtol:
        real number, default = 1e-5 - The relative difference tolerance
            for equality. See numpy.allclose.
    atol:
        real number, default = 1e-8 - The absolute tolerance parameter
            for equality. See numpy.allclose.
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


    def get_feature_names_out(self):
        # pizza!
        # when there is a {'Intercept': 1} in :param: keep, need to make sure
        # that that column is accounted for here! and the dropped columns are
        # also accounted for!
        pass


    def get_metadata_routing(self):
        """
        Get metadata routing is not implemented in ColumnDeduplicateTransformer.

        """
        __ = type(self).__name__
        raise NotImplementedError(
            f"get_metadata_routing is not implemented in {__}"
        )


    # def set_params(self):
        # pizza! dont forget! once the instance is fitted, cannot change equal_nan, rtol, and atol!
        # ... or maybe u can.... its just that new fits will be fitted subject to
        # different rules than prior fits


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
        # validation of X must be done here (with reset=True), not in a
        # separate module
        # BaseEstimator has _validate_data method, which when called
        # exposes n_features_in_ and feature_names_in_.
        X = self._validate_data(
            X=X,
            reset=not hasattr(self, "duplicates_"),
            cast_to_ndarray=False,
            # vvv become **check_params, and get fed to check_array()
            accept_sparse=("csr", "csc", "coo", "dia", "lil", "dok"),
            dtype=None,
            force_all_finite="allow-nan",
            ensure_2d=True,
            ensure_min_features=2,
            order='F'
        )

        # reset – Whether to reset the n_features_in_ attribute. If False,
        # the input will be checked for consistency with data provided when
        # reset was last True.
        # It is recommended to call reset=True in fit and in the first call
        # to partial_fit. All other methods that validate X should set
        # reset=False.
        #
        # cast_to_ndarray – Cast X and y to ndarray with checks in
        # check_params. If False, X and y are unchanged and only
        # feature_names_in_ and n_features_in_ are checked.

        # ^^^^^^ pizza be sure to review _validate_data! ^^^^^^^

        _validation(
            X,
            self.feature_names_in_ if hasattr('self', 'feature_names_in_') else None,
            self.keep,
            self.equal_nan,
            self.rtol,
            self.atol,
            self.n_jobs
        )



        # dictionary of column indices and respective constant values
        self.constant_columns_:dict[int, any] = \
            _find_constants(
                X,
                self.constant_columns_ if hasattr(self, 'constant_columns_') else {},
                self.equal_nan,
                self.rtol,
                self.atol,
                self.n_jobs
            )


        # pizza head, take it easy on yourself! instead of passing _keep
        # callable and _X to _make_instructions, calculate it here and just
        # send an int!

        # note to future pizza, remember that in runtime once the callable
        # returns the index, validate that the column actually is constant
        if callable(self.keep):
            _keep = self.keep(X)
            if not _keep in self.constant_columns_:
                raise ValueError(
                    f"'keep' callable has returned an integer column index "
                    f"that is not a column of constants. \nconstant columns: "
                    f"{self.constant_columns_}"
                )
        else:
            _keep = self.keep



        _instructions = _make_instructions(
            _keep,
            self.constant_columns_,
            self.feature_names_in_ if hasattr(self, 'feature_names_in_') else None,
            X.shape
        )

        self.kept_columns_, self.removed_columns_, self.column_mask_ = \
            _set_attributes(
                self.constant_columns_,
                _instructions,
                self.n_features_in_
            )






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
        _instructions = _make_instructions(
            self.keep,
            self.constant_columns_,
            self.feature_names_in_ if hasattr(self, 'feature_names_in_') else None,
            X.shape
        )

        self.kept_columns_, self.removed_columns_, self.column_mask_ = \
            _set_attributes(
                self.constant_columns_,
                _instructions,
                self.n_features_in_
            )


        out = _transform(X, _instructions)

        # Pizzahead remember to do the 'C' order setting trick that
        # u did in CDT

        return out


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












