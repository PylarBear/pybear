# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import numpy as np
import numpy.typing as npt
import pandas as pd


import numbers
from typing import Iterable, Literal, Optional
from typing_extensions import Union, Self

from numbers import (Integral, )



from sklearn.base import BaseEstimator, TransformerMixin, _fit_context
from sklearn.exceptions import NotFittedError
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_is_fitted, check_array



class NoDupPolyFeatures(BaseEstimator, TransformerMixin):

    """
    make pizza


    Parameters
    ----------


    Attributes
    ----------
    n_features_in_:
        int - number of features in the fitted data before deduplication.

    feature_names_in_:
        NDArray[str] - The names of the features as seen during fitting.
        Only accessible if X is passed to :methods: partial_fit or fit
        as a pandas dataframe that has a header.

    duplicates_: list[list[int]] - a list of the groups of identical
        columns, indicated by their zero-based column index positions
        in the originally fit data.

    removed_columns_: dict[int, int] - a dictionary whose keys are the
        indices of duplicate columns removed from the original data,
        indexed by their column location in the original data; the values
        are the column index in the original data of the respective
        duplicate that was kept.

    column_mask_: list[bool], shape (n_features_,) - Indicates which
        columns of the fitted data are kept (True) and which are removed
        (False) during transform.

    "drop_constants": ["boolean"],
    # pizza!



    Notes
    -----


    See Also
    --------


    Examples
    --------






    """

    _parameter_constraints: dict = {
        "degree": [Interval(Integral, 0, None, closed="left")],
        "min_degree": [Interval(Integral, 0, None, closed="left")],
        "keep": [StrOptions({"first", "last", "random"})],
        "do_not_drop": [list, tuple, set, None, ],
        "conflict": [StrOptions({"raise", "ignore"})],
        "drop_duplicates": ["boolean"],
        "interaction_only": ["boolean"],
        "include_bias": ["boolean"],
        "drop_constants": ["boolean"],
        "output_sparse": ["boolean"],
        "order": [StrOptions({"C", "F"})],
        "rtol": [numbers.Real],
        "atol": [numbers.Real],
        "equal_nan": ["boolean"],
        "n_jobs": [numbers.Integral, None],
    }


    def __init__(
        self,
        degree:int=2,
        *,
        min_degree:int=0,
        drop_duplicates: Optional[bool] = True,
        keep: Optional[Literal['first', 'last', 'random']] = 'first',
        do_not_drop: Optional[Union[Iterable[int], Iterable[str], None]] = None,
        conflict: Optional[Literal['raise', 'ignore']] = 'raise',
        interaction_only: Optional[bool] = False,
        include_bias: Optional[bool] = True,
        drop_constants: Optional[bool] = True,
        output_sparse: Optional[bool] = False,
        order: Optional[Literal['C', 'F']] = 'C',
        rtol: Optional[float] = 1e-5,
        atol: Optional[float] = 1e-8,
        equal_nan: Optional[bool] = False,
        n_jobs: Optional[Union[int, None]] = None
    ):

        pass


    def get_feature_names_out(
        self,
        input_features: Union[Iterable[str], None]=None
    ) -> npt.NDArray[str]:

        """

        Parameters
        ----------


        Return
        ------



        """
        pass



    # def get_params(self, deep=?:bool) -> dict[str: any]:
    # if ever needed, hard code that can be substituted for the
    # BaseEstimator get/set_params can be found in GSTCV_Mixin


    def partial_fit(
        self,
        X: PIZZA,
        y: Union[Iterable[any], None]=None
    ) -> Self:
        """

        Parameters
        ----------




        Return
        ------


        """
        pass


    def fit(
        self,
        X: PIZZA,
        y: Union[Iterable[any], None]=None
    ) -> Self:
        """

        Parameters
        ----------




        Return
        ------


        """

        return self.partial_fit(X, y)


    # def set_params
    # if ever needed, hard code that can be substituted for the
    # BaseEstimator get/set_params can be found in GSTCV_Mixin


    def score(
        self,
        X,
        y:Union[Iterable[any], None]=None
    ) -> None:
        """
        Dummy method to spoof dask Incremental and ParallelPostFit
        wrappers. Verified must be here for dask wrappers.
        """

        pass


    def transform(
        self,
        X: PIZZA
    ) -> PIZZA!:

        """



        Parameters
        ----------


        Return
        -------


        """

        pass































